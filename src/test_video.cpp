#include "zed_fusion_perception/yolo26_tensorrt.hpp" // Adatta il path al tuo progetto
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Utilizzo: " << argv[0] << " <path_engine> <path_video_input>"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string video_path = argv[2];

  try {
    // 1. Inizializza il modello
    Yolo26nSeg yolo(engine_path, 0.45, 0.45); // Engine, Conf_thresh, NMS_thresh

    // 2. Apri il video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::cerr << "Errore: Impossibile aprire il video: " << video_path
                << std::endl;
      return -1;
    }

    // 3. Prepara il VideoWriter per salvare il risultato
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("output_result.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));

    std::cout << "Inizio elaborazione video (" << width << "x" << height
              << " @ " << fps << "fps)..." << std::endl;

    cv::Mat frame;
    int frame_count = 0;

    while (cap.read(frame)) {
      // --- INFERENZA ---
      auto start = std::chrono::high_resolution_clock::now();
      auto detections = yolo.infer(frame);
      auto end = std::chrono::high_resolution_clock::now();

      // --- VISUALIZZAZIONE ---
      for (const auto &det : detections) {
        // Scegli il colore in base alla classe (0: Blu, 1: Giallo, 2: Arancio)
        cv::Scalar color;
        if (det.class_id == 0)
          color = cv::Scalar(255, 0, 0); // Blue
        else if (det.class_id == 1)
          color = cv::Scalar(0, 255, 255); // Yellow
        else if (det.class_id == 2)
          color = cv::Scalar(0, 165, 255); // Orange
        else
          color = cv::Scalar(0, 255, 0);

        // Disegna centro
        cv::circle(frame, det.center_2d, 4, color, -1);

        // Disegna Maschera se presente
        if (!det.mask.empty()) {
          cv::Mat mask_color;
          cv::cvtColor(det.mask, mask_color, cv::COLOR_GRAY2BGR);
          mask_color.setTo(color, det.mask);
          cv::addWeighted(frame, 1.0, mask_color, 0.4, 0, frame);
        }
      }

      // Scrivi il frame elaborato
      writer.write(frame);

      // Log di progresso
      double ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      if (frame_count % 30 == 0) {
        std::cout << "Frame: " << frame_count << " | Tempo Inferenza: " << ms
                  << "ms" << std::endl;
      }
      frame_count++;
    }

    std::cout
        << "Elaborazione completata! Risultato salvato in: output_result.mp4"
        << std::endl;
    cap.release();
    writer.release();

  } catch (const std::exception &e) {
    std::cerr << "ECCEZIONE: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}