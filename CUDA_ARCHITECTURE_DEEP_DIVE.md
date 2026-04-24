# Deep Dive: Architettura della Pipeline CUDA per la Percezione ad Alte Prestazioni

Questo documento analizza le scelte architetturali effettuate nell'implementazione CUDA di questo nodo, con un focus specifico sulla gestione della memoria, sulla località dei dati e sull'ottimizzazione del throughput della cache.

## 1. Il Problema: Layout di Memoria CHW e Cache Miss
I modelli di Deep Learning come YOLO26 producono mappe di prototipi (mask prototypes) in formato **CHW** (Channel-Height-Width). In questo layout, i valori dei canali per un singolo pixel spaziale $(x, y)$ sono distribuiti in memoria a una distanza pari a $H \times W$.

### 1.1 Analisi dell'Accesso Non-Contiguo
Durante la fase di post-processing (generazione della maschera), per ogni pixel $(x, y)$ della canvas, dobbiamo calcolare un prodotto scalare tra 32 coefficienti di detection e 32 valori di prototipo:
$$Mask(x, y) = \sigma \left( \sum_{c=0}^{31} coeff_c \times proto(x, y, c) \right)$$

In formato **CHW**, accedere a 32 canali per lo stesso pixel significa caricare 32 word di memoria separate da migliaia di byte. Per un'architettura GPU (come Turing o Ampere), questo causa:
1. **Cache Miss Sistematici**: Ogni thread del warp richiede dati che non si trovano nella stessa linea di cache L1/L2 dei thread adiacenti.
2. **Memory Divergence**: Il controller di memoria non può effettuare il "coalescing" delle richieste, saturando la banda del bus PCIe/VRAM con transazioni inefficienti.

## 2. La Soluzione: Reformat Kernel (CHW to HWC)
Per risolvere il problema alla radice, abbiamo introdotto un kernel di **trasposizione hardware-accelerata** che riorganizza i dati in formato **HWC** (Height-Width-Channel).

### 2.1 Perché "Girare la Matrice" Garantisce Dati Contigui
In formato **HWC**, i 32 canali del pixel $(x, y)$ sono memorizzati in posizioni di memoria **fisicamente adiacenti**. 
- Indirizzo in CHW: $Base + c \times (H \times W) + y \times W + x$ (Distanti!)
- Indirizzo in HWC: $Base + (y \times W + x) \times 32 + c$ (Contigui!)

Quando un thread CUDA legge il primo canale del pixel, l'intero blocco di 32 canali viene caricato nella **L1 Cache/Shared Memory** in un'unica operazione di memoria (o in pochissime transazioni coalesced). Questo riduce drasticamente la latenza di caricamento dei dati per il calcolo del prodotto scalare.

## 3. Ottimizzazione della Shared Memory Tiling
Nel kernel `postprocess_mask_kernel_optimized`, non ci limitiamo alla contiguità dei prototipi. Utilizziamo la **Shared Memory** (una memoria on-chip ad altissima velocità, simile a un registro L0 programmabile) per memorizzare i bboxes e i coefficienti delle 128 detection più rilevanti.

### 3.1 Eliminazione delle Letture Ridondanti
Senza Shared Memory, ogni thread (uno per ogni pixel della canvas, es. 2 milioni di thread per un'immagine Full HD) dovrebbe leggere i 32 coefficienti dalla memoria globale per ogni detection.
Utilizzando il pattern di **Tiling**:
1. All'inizio del blocco CUDA, i thread cooperano per caricare i coefficienti e le bboxes dalla Global Memory alla Shared Memory.
2. Una volta sincronizzati (`__syncthreads()`), tutti i thread del blocco accedono ai coefficienti con latenza quasi nulla.
3. Il numero di accessi alla Global Memory passa da $O(Pixels \times Detections)$ a $O(Blocks \times Detections)$.

## 4. Analisi sulla Rimozione dei CUDA Graphs
Inizialmente, avevamo implementato i **CUDA Graphs** per ridurre l'overhead di lancio dei kernel (launch overhead). Tuttavia, in un sistema di percezione real-time dove:
- La latenza del kernel TensorRT domina l'intero frame (~8-10ms).
- L'overhead di lancio di 4 kernel è nell'ordine dei microsecondi ($< 0.1\%$ del totale).
- La flessibilità del cambio dinamico di indirizzi di memoria (canvas, stream) è prioritaria.

Abbiamo deciso di rimuoverli per ridurre la complessità del codice e migliorare la stabilità della pipeline asincrona. La perdita di performance è teoricamente non misurabile su architetture moderne, mentre il guadagno in manutenibilità è significativo.

## 5. Conclusioni Architetturali
L'efficienza di questa pipeline non deriva dalla "forza bruta" del calcolo, ma dalla **simmetria tra layout di memoria e gerarchia delle cache**. Trattare la GPU non come un processore generico, ma come un acceleratore guidato dal throughput di memoria, ci ha permesso di ottenere frequenze di processing superiori ai 100Hz su hardware entry-level (NVIDIA T1000), garantendo al contempo una latenza deterministica per il nodo di fusione.
