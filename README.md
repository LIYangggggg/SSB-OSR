# [ECCV 2024 OOD-CV Workshop SSB Challenge (Open-Set Recognition Track) ](https://www.ood-cv.org/challenge.html)- 1st Place

## Environment Setup

To replicate the environment, please ensure you have Python installed. The required dependencies are listed in the `requirements.txt` file. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

## Training

To train the model, use the following script:

```bash
bash run_deit_Inet1k.sh
```

This script will initiate the training process using the pre-configured optimal parameters and the [official pretrained model](https://drive.google.com/file/d/1Y4DmcHhngex6h8B6bX-1Ehl6i6dJVgO7/view?usp=drive_link) for fine-tuning. The resulting trained model will be saved in the specified directory.


## Testing

After training, you can evaluate the model using the following script. We provide fine-tuned model [DeiT III-Base](https://drive.google.com/file/d/1mpiZn1GP3K08L_RKjndI3WudceM53cUK/view?usp=sharing) and have set optimal default parameters for both training and testing. 

```bash
bash run_eval_TTA.sh
```

This will generate multiple prediction results.

To fuse the prediction results and print them, use the following script:

```bash
python metric_result_GradNorm.py --result_dir path/to/
```

### Results

The table below shows the competition results, with **Intellindust-AI-Lab (ours)** achieving the best average performance:

| Team                    | AUROC | FPR@TPR95 |
|-------------------------|-------|-----------|
| **Intellindust-AI-Lab (ours)**  | **81.54**  | 61.72   |
| wangzhiyu918            | 80.82 | 67.37     | 
| xuanxuan                | 79.94 | 65.86     | 
| WJB                     | 79.84 | 66.05     | 
| aesdfdgfgff             | 79.77 | **61.44**     |


