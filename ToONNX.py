import hydra
from transformers import AutoTokenizer
import pandas as pd
from utils.text_cleaner import Cleaner
from models.bert_model import BertModel
import pickle
import torch
import os


@hydra.main(config_path="conf/",config_name="config")
def main(cfg):

    # Read configs
    inference_conf = cfg['inference_conf']
    model_conf = cfg['model_conf']

    # Read mode,tokenizer and label encoder
    label_encoder =  pickle.load(open(inference_conf['label_map_path'], "rb"))
    num_labels = len(label_encoder.classes_)
    model = BertModel(model_conf,num_labels,inference_flag=True)
    
    # Save Model To ONNX

    x = torch.randn(1, 32, requires_grad=True).type(torch.LongTensor).to('cuda')
    y = torch.randn(1, 32, requires_grad=True).type(torch.LongTensor).to('cuda')
    torch.nn.functional.relu(x, inplace=True)
    torch.nn.functional.relu(y, inplace=True)

    with torch.no_grad():
        torch_out = model.model(x.to('cuda'), y.to('cuda'))
        torch_out = torch_out.logits.detach().cpu().numpy()
    
    print(torch_out)
    
     ## Save label map
    if not os.path.isdir(model_conf['onnx_path']):
        os.mkdir(model_conf['onnx_path'])
        
    torch.onnx.export(model.model,               # model being run
                  (x,y),                         # model input (or a tuple for multiple inputs)
                  f"{model_conf['onnx_path']}/dynamic_saved_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input_ids', 'attention_mask'],   # the model's input names
                  output_names = ['softmax'], # the model's output names
                  dynamic_axes={'input_ids' : {0 : 'batch_size'},    # variable length axes
                                'attention_mask' : {0 : 'batch_size'}, 
                                'softmax' : {0 : 'batch_size'}})

if __name__ == "__main__":
    main()

# !python ToONNX.py


