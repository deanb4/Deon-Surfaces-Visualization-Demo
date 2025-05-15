import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_data
import numpy as np
from PIL import Image


def preprocess(batch, feature_extractor):
    inputs = feature_extractor(images=batch["image"])
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}

    mask = batch["segmentation_mask"]
    mask = Image.fromarray(mask).resize(feature_extractor.size, resample=Image.NEAREST)
    labels = torch.tensor(np.array(mask), dtype=torch.long)

    inputs["labels"] = labels
    return inputs


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    valid = labels != -100
    accuracy = (preds[valid] == labels[valid]).sum() / valid.sum()
    return {"pixel_accuracy": accuracy}


def main():
    dataset = load_data("wyrx/SUNRGBD_seg", "uint8")
    print(dataset)

    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True)
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

    # Use lambda or partial to pass feature_extractor to preprocess
    train_dataset = dataset["train"].map(lambda batch: preprocess(batch, feature_extractor), remove_columns=dataset["train"].column_names)
    val_dataset = dataset["validation"].map(lambda batch: preprocess(batch, feature_extractor), remove_columns=dataset["validation"].column_names)

    training_args = TrainingArguments(
        output_dir="./segformer-b5-sunrgbd",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=100,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    )

    trainer.train()

    trainer.save_model("./segformer-b5-sunrgbd-finetuned")
    feature_extractor.save_pretrained("./segformer-b5-sunrgbd-finetuned")


if __name__ == "__main__":
    main()
