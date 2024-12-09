from ultralytics import YOLO
import torch
import json

def train_model():
    # Path to the YAML configuration file for training
    data_yaml = './data.yaml'
    model = YOLO('yolov10m.pt')

    epochs=250
    imgw=960
    imgh=540
    batch=8
    #Learning Rate
    lr0=0.00001
    #Final Learning Rate
    lrf=0.0000001
    optimizer='AdamW'
    workers=4
    confidence=0.1
    iou=0.5
    
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        imgsz=(imgw, imgh), 
        batch=batch, 
        name='v10M-' + str(epochs) + 'E-' + str(imgw) + 'x' + str(imgh) + '-C' + str(confidence) + '-B' + str(batch) + '-IOU' + str(iou) + '-LR' + str(lr0) + '-' + str(lrf), 
        save=True, 
        augment=True,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        workers=workers
    )

    # Evaluate the model performance on the validation set with a confidence threshold of 0.2
    metrics = model.val(conf=confidence, iou=iou)
    print(metrics)

    precision = metrics.results_dict.get('metrics/precision(B)', 0)
    recall = metrics.results_dict.get('metrics/recall(B)', 0)
    mAP50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    mAP50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
    
    # Calculate F1-score
    accuracy = (2*(precision * recall)) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP50: {mAP50}")
    print(f"mAP50-95: {mAP50_95}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    results_dict = metrics.results_dict
    save_dir = metrics.save_dir
    speed = metrics.speed
    task = metrics.task

    output_data = {
        "results_dict": results_dict,
        "precision": precision,
        "recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "accuracy": accuracy,  # Add accuracy (F1-Score) to the output data
        "save_dir": str(save_dir),  # Convert Path object to string
        "speed": speed,
        "task": task
    }

    # Write the results to a file
    with open('training_results.json', 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == '__main__':
    train_model()
    torch.cuda.empty_cache()
