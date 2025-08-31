#!/bin/bash

# 顺序执行三个Python脚本

# 设置日志文件
LOG_FILE="/data/zhangxiaofeng/code/code/spectomol/sequential_training.log"
echo "$(date) - 开始执行顺序训练任务" >> $LOG_FILE

# 脚本路径
SCRIPT1="/data/zhangxiaofeng/code/code/spectomol/train_featurefusion.py"
SCRIPT2="/data/zhangxiaofeng/code/code/spectomol/train_mlp.py"
SCRIPT3="/data/zhangxiaofeng/code/code/spectomol/train_transformer.py"

# 运行第一个脚本
echo "$(date) - 开始训练模型1" >> $LOG_FILE
echo "$(date) - 开始训练模型1"
python $SCRIPT1
if [ $? -ne 0 ]; then
    echo "$(date) - 模型1训练失败,退出程序" >> $LOG_FILE
    echo "$(date) - 模型1训练失败,退出程序"
    exit 1
fi
echo "$(date) - 模型1训练完成" >> $LOG_FILE
echo "$(date) - 模型1训练完成"

# 运行第二个脚本
echo "$(date) - 开始训练模型2" >> $LOG_FILE
echo "$(date) - 开始训练模型2"
python $SCRIPT2
if [ $? -ne 0 ]; then
    echo "$(date) - 模型2训练失败,退出程序" >> $LOG_FILE
    echo "$(date) - 模型2训练失败,退出程序"
    exit 1
fi
echo "$(date) - 模型2训练完成" >> $LOG_FILE
echo "$(date) - 模型2训练完成"

# 运行第三个脚本
echo "$(date) - 开始训练模型3" >> $LOG_FILE
echo "$(date) - 开始训练模型3"
python $SCRIPT3
if [ $? -ne 0 ]; then
    echo "$(date) - 模型3训练失败,退出程序" >> $LOG_FILE
    echo "$(date) - 模型3训练失败,退出程序"
    exit 1
fi
echo "$(date) - 模型3训练完成" >> $LOG_FILE
echo "$(date) - 模型3训练完成"

echo "$(date) - 所有模型训练完成！" >> $LOG_FILE
echo "$(date) - 所有模型训练完成！"