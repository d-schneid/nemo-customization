**Command:** `srun --ntasks-per-node=1 --gres=gpu:1 --mem-per-gpu=100G --container-name=nemo_2502_$USER --container-mounts=/shared:/shared python /shared/home/$USER/customization_simple/train.py`

**Different errors depending on configuration:**

**Error 1:**

![image](https://github.com/user-attachments/assets/93dff142-f2b0-494e-a0bc-57039589a008)

**Error 2:**

![Screenshot 2025-04-28 at 13 16 52](https://github.com/user-attachments/assets/26458bc6-56e1-4703-9f07-d3e91c1fd622)

