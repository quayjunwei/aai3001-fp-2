import os

print("Current working directory:", os.getcwd())
print("Does the file exist?", os.path.exists("../models/densenet121_epoch55.pth"))
