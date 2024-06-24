import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: It doesn't support multiple behaviors in one env. Need to find a neat way to load multiple checkpoints. I should save checkpoints with dicts that hold all behaviors in one checkpoint


env = UnityInterface("Builds\\Windows\\Ball3D\\UnityEnvironment")      # 1D obs only, continuous action of size 2. Rewards: 0.1 for every step, -1 for fail, 100 is the max episodic return
# env = UnityInterface("Builds\\Windows\\Crawler\\UnityEnvironment")     # 1D obs only, continuous action of size 8
# env = UnityInterface("Builds\\Windows\\PushBlock\\UnityEnvironment")   # 1D obs only, discrete action of size (7). Rewards: 5 for win, -0.001 for every step
# env = UnityInterface("Builds\\Windows\\WallJump\\UnityEnvironment")    # 1D obs only, discrete action of size (3, 3, 3, 2)
# env = UnityInterface(None) 

print(f"{env.getSpecs()}")
behaviorNames = env.getBehaviorNames()

checkpointNames = ["3DBall--5000", "3DBall--10000", "3DBall--15000", "3DBall--30000", "3DBall--50000", "3DBall--100000", "3DBall--150000"]
checkpoints = [f"checkpoints\\{name}.pth" for name in checkpointNames]

for i, checkpoint in enumerate(checkpoints):
    actor = {}
    for behavior in behaviorNames:
        envSpecs = env.getSpecs(behavior)
        actor[behavior] = SAC(envSpecs).to(device)
        checkpointLoaded = torch.load(checkpoint)
        actor[behavior].load_state_dict(checkpointLoaded["actor"])
        exportONNX(f"onnx\\{checkpointNames[i]}", actor[behavior], env.getSpecs(behavior))

env.close()