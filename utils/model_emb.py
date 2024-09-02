import torch 
from collections import OrderedDict


def model_emb(model_paths, save_path):
    pass
    model_static_dict_list = []
    
    for model_path in model_paths:
        model = torch.load(model_path, map_location='cpu')
        model_static_dict_list.append(model)
    
    emb_model = OrderedDict()
    for k in model_static_dict_list[0].keys():

        mean_val = 0
        for model_dict in model_static_dict_list:
            if k not in model_dict:
                assert f"Can't find k:{k}"
            mean_val += model_dict[k]
        mean_val = mean_val/len(model_static_dict_list)

        emb_model[k] = mean_val
    
    torch.save(emb_model, save_path)



if __name__ == "__main__":

    model_paths = [

        "./baseline/imagenet_fmfp-mixup_0.2-crl_0.0/best_acc_net.pth",
        "./res384ep30/imagenet_fmfp-mixup_0.2-crl_0.0/best_acc_net.pth",
    ]

    save_path = r"./best_acc_net.pth"
    model_emb(model_paths, save_path)

