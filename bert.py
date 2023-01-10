#https://habr.com/ru/company/rostelecom/blog/704844/

import numpy as np
import pandas as pd
import torch, onnx
import time
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)

import onnxruntime


def convert_from_torch_to_onnx(onnx_path: str, tokenizer:AutoTokenizer, model:AutoModelForSequenceClassification) -> None:
    dummy_model_input = tokenizer(
        "текст для конвертации",
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cpu")
    torch.onnx.export(
        model,
        dummy_model_input["input_ids"],
        onnx_path,
        opset_version=12,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_len"
            },
            "output": {
                0: "batch_size"
            }
        }
    )

def convert_from_onnx_to_quantized_onnx(
        onnx_model_path: str,
        quantized_onnx_model_path: str
) -> None:
    """Квантизация модели в формате ONNX до Int8
    и сохранение кванитизированной модели на диск.

    @param onnx_model_path: путь к модели в формате ONNX
    @param quantized_onnx_model_path: путь к квантизированной модели
    """
    quantize_dynamic(
        onnx_model_path,
        quantized_onnx_model_path,
        weight_type=QuantType.QUInt8
    )

if __name__ == "__main__":
    model_name = "ElKulako/cryptobert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64,
                                              truncation=True, padding='max_length', top_k=None)

    news_list = [
        "How the GBTC premium commerce ruined Barry Silbert, his DCG empire and took crypto lending platforms with them",
        "MicroStrategy’s Bitcoin Strategy Sets Tongues Wagging Even As It Doubles Down On BTC Purchases ⋆ ZyCrypto",
        "How the GBTC premium trade ruined Barry Silbert, his DCG empire and took crypto lending platforms with them",
        "Bitcoin Holders To Expect More Difficulties As Data Point To Looming BTC Price Drop",
        "Bitcoin Breaks Past $17,000 Barrier – Will BTC Also Breach 4% Weekly Run?",
        "Bitcoin Price Today 9 Jan: BTC Increases By 1.79% Painting The Chart Green",
        "Bitcoin: This is what large investor and retail interest can do for BTC over time"
        ]
    start_time = time.time()
    torch_out = pipe(news_list[0])

    for n in news_list:
        input_ids = torch.tensor(tokenizer.encode(n)).unsqueeze(0)
        outputs = model(input_ids)
        print(outputs)
    print(time.time()-start_time)

    # start_time = time.time()
    # convert_from_torch_to_onnx("bert.onnx", tokenizer, model)
    # print(time.time() - start_time)
    #
    #
    #
    # onnx_model = onnx.load("bert.onnx")
    # onnx.checker.check_model(onnx_model)
    #
    #
    # ort_session = onnxruntime.InferenceSession("bert.onnx")
    #
    #
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #
    #
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    #
    # print("finish")
    #
