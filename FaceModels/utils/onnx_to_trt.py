import os
import tensorrt as trt
import sys
from typing import Tuple, Union
import logging

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def _build_engine_onnx(input_onnx:Union[str, bytes], force_fp16:bool=False, 
                       max_batch_size:int=1, max_workspace:int=1024):
    """
    build tensorrt engine from provided ONNX file
    :param input_onnx: serialized onnx model
    :param force_fp16: force use of fp16 precision, even if device doesn't support
    :param max_batch_size: define maximum batch size supported by engine. If > 1, creates optimization profile
    :param max_workspace: Maximum builder workspace in MB
    :return: TensorRT engine
    """
    
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        if force_fp16 is True:
            logging.info("building tensorrt engine with FP16 support.")
            has_fp16 = builder.platform_has_fast_fp16
            if not has_fp16:
                logging.warnings("Builder report no fast FP16 support. ")
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            
        config.max_workspace_size = max_workspace * 1024 * 1024
        
        if not parser.parse(input_onnx):
            print("ERROR: Failed to parse the ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)
            
        if max_batch_size != 1:
            logging.warning("Batch size != 1 is used. Ensure your inference code supports it.")
        profile = builder.create_optimization_profile()
        #get input name and shape for building optimization profile
        input = network.get_input(0)
        im_size = input.shape[2:]
        input_name = input.name
        profile.set_shape(input_name, (1, 3) + im_size, (1, 3) + im_size,
                          (max_batch_size, 3) + im_size)
        config.add_optimization_profile(profile)
        return builder.build_engine(network, config=config)
    
def convert_onnx(input_onnx:Union[str, bytes], engine_file_path:str, 
                 force_fp16:bool=False, max_batch_size:int=1):
    """
    creates tensorrt engine and serializes it to disk
    :param input_onnx: path to onnx file on disk or serialized onnx model
    :param engine_file_path: path where tensorrt engine should be saved
    :param force_fp16: foce use of fp16 precision, even if device doesn't support it
    :param max_batch_size: define maximum batch size supported by engine
    :return: None
    """
    onnx_obj = None
    if isinstance(input_onnx, str):
        with open(input_onnx, "rb") as f:
            onnx_obj = f.read()
    elif isinstance(input_onnx, bytes):
        onnx_obj = input_onnx
    
    engine = _build_engine_onnx(input_onnx=onnx_obj, force_fp16=force_fp16, 
                                max_batch_size=max_batch_size)
    assert not isinstance(engine, type(None))
    
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    
