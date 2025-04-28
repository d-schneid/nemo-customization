I try with efa-enabled and not efa-enabled container: used-container=nemo_efa_2502 | nemo_2502

**Command:** `srun --ntasks-per-node=1 --gres=gpu:1 --mem-per-gpu=100G --container-name=used-container --container-mounts=/shared:/shared python /shared/home/$USER/customization_simple/train.py`

**Different errors depending on configuration:**

**Error with used-container=nemo_2502:**

![Screenshot 2025-04-28 at 15 26 09](https://github.com/user-attachments/assets/435049a6-e201-46d3-8e83-ae0f43716b3b)

**Error with used-container=nemo_efa_2502:**

![Screenshot 2025-04-28 at 13 28 35](https://github.com/user-attachments/assets/6b9e7171-f718-4298-beb2-c5bf2c5f5be2)

**Previously Encountered Error:**

![image](https://github.com/user-attachments/assets/93dff142-f2b0-494e-a0bc-57039589a008)

**Hardware:**

![Screenshot 2025-04-28 at 12 39 03](https://github.com/user-attachments/assets/aaf98fed-91d6-47c3-872a-ca8db9a65bd5)


![Screenshot 2025-04-28 at 12 52 59](https://github.com/user-attachments/assets/e51de9a1-2eb1-4a1a-aeac-2a46c1657d97)
