import os
import json
import torch
import argparse
import lightning as L

from lightning.pytorch.callbacks import ModelSummary
from train import SkinCancerDataModule, SkinCancerModule
from sklearn.metrics import confusion_matrix, classification_report


def main(args):
    dm = SkinCancerDataModule(
        batch_size=args.batch_size, 
        img_size=args.img_size,
        train_dir="./train", 
        test_dir=args.test_dir
    )
    dm.prepare_data()
    dm.setup('test')
    
    model = SkinCancerModule.load_from_checkpoint(
        checkpoint_path=args.model,
    )

    model.eval()
    model.freeze()

    trainer = L.Trainer(enable_progress_bar=True, enable_model_summary=True)

    summary_callback = ModelSummary(max_depth=2)
    trainer.callbacks.append(summary_callback)
    trainer.fit_loop.epoch_loop.refit(model, dm)  

    test_results = trainer.test(model, dm.test_dataloader(), verbose=True)

    class_names = dm.trainData.classes

    y_true = []
    y_pred = []
    for batch in dm.test_dataloader():
        x, y = batch
        with torch.no_grad():
            output = model(x)
            _, predicted = torch.max(output, 1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())


    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    results = {
        "test_accuracy": test_results[0]['test_acc'],
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "class_names": class_names,
        "num_test_samples": len(dm.testData),
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "model_checkpoint": args.model
    }

    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default="./model.ckpt", help='Path to the model checkpoint')
    parser.add_argument('--test_dir', type=str, default='./test', help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save evaluation results')

    args = parser.parse_args()

    main(args)