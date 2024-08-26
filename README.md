以下是更新后的README内容，其中包含超链接指向预训练模型的下载地址，并指定模型为DeiTIII-Base：

```markdown
# SSB-OSR

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

This script will initiate the training process using the pre-configured optimal parameters. The resulting trained model will be saved in the specified directory.

## Testing

After training, you can evaluate the model using the following script:

```bash
bash run_eval_TTA.sh
```

This will generate multiple prediction results, which will be saved in the `save_dir` directory.

## Result Fusion and CSV Testing

To fuse the prediction results and export them to a CSV file, use the following script:

```bash
python metric_result_GradNorm.py
```

This script will process the saved prediction results and produce a final CSV file containing the fused results. After generating the CSV file, you can further test the results by using the following script:

```bash
python metric_csv.py
```

This script will evaluate the CSV file and provide the final test results.

## Pre-trained Models and Default Parameters

We provide pre-trained model [DeiT III-Base](https://drive.google.com/file/d/1mpiZn1GP3K08L_RKjndI3WudceM53cUK/view?usp=sharing) and have set optimal default parameters for both training and testing. 

### Results

The table below shows the competition results, with **Intellindust-AI-Lab (ours)** achieving the best average performance:

| Team                    | AUROC | FPR@TPR95 |
|-------------------------|-------|-----------|
| **Intellindust-AI-Lab (ours)**  | **81.54**  | 61.72   |
| wangzhiyu918            | 80.82 | 67.37     | 
| xuanxuan                | 79.94 | 65.86     | 
| WJB                     | 79.84 | 66.05     | 
| aesdfdgfgff             | 79.77 | **61.44**     |
```

