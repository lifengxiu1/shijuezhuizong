# 1. 创建名为 lingvoo 的 conda 环境，指定 Python 版本
conda create -n lingvoo python=3.10 -y

# 2. 激活这个环境
conda activate lingvoo

# 3. 安装依赖
# numpy 和 opencv 用 conda 装（更稳定），mediapipe 用 pip 装
conda install -c conda-forge numpy opencv -y
pip install mediapipe

# 3. 安装Lerobot
pip install 'lerobot[so100]'

# 4. 安装 LeRobot 适配的 Feetech SDK 版本
    pip install "feetech-servo-sdk==1.0.0"
    测试
    python -c "import scservo_sdk as scs; print('scservo_sdk from feetech-servo-sdk:', hasattr(scs, 'PacketHandler'))"
# 5. 校准机械臂
    1.启动校准
    conda activate lingvoo
    lerobot-calibrate --robot.type=so100_follower --robot.port=COM3
    具体操作步骤
    2.摆到“中间舒适位”，再按一次回车
    看到提示：
        Move SO100Follower to the middle of its range of motion and press ENTER....
    这里要做的是：
        用手轻轻拖着每个关节，把整只手臂摆成一个你觉得“中立、不顶到任何边界”的姿势
            不是完全伸直
            也不是缩成一团
            就是一个“准备开始工作”的中间姿态
    然后在键盘上按一次 Enter 回车。
    3.接下来会出现：
        Move all joints except 'wrist_roll' sequentially through their entire ranges of motion.
        Recording positions. Press ENTER to stop...
        在按回车之前，请按顺序 一个关节一个关节慢慢转满行程，尤其是：
            shoulder_pan（肩水平转）
            shoulder_lift（肩上下）
            elbow_flex（肘弯曲）
            wrist_flex（腕上下）
            gripper（夹爪开合）
        具体动作建议：
            拿稳底座，让手臂别撞桌子；
            拿着机械臂末端或相应关节，轻柔地：
            从一侧极限 → 转到另一侧极限
            来回 2～3 次
            轮到 wrist_flex 和 gripper 的时候：
            腕关节：让手腕尽量往上弯，再尽量往下弯（别暴力，用一点点力慢慢推）
            夹爪：完全张开 → 完全夹紧，来回几次
        ⚠ 很重要：
            只有在你觉得每个关节都已经“跑了几遍全程之后”，再去按回车。
    4.看一下新的表格是否正常
        NAME | MIN | POS | MAX 
        shoulder_pan | 1297 | 1981 | 3048 
        shoulder_lift | 1031 | 2004 | 3054 
        elbow_flex | 1913 | 2147 | 4001 
        wrist_flex | 1072 | 2126 | 3146 
        gripper | 2037 | 2044 | 3386
# 6. 启动
    python gesture_cam.py
