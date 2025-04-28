I try with efa-enabled and not efa-enabled container: used-container=nemo_efa_2502 | nemo_2502

**Command:** `srun --ntasks-per-node=1 --gres=gpu:1 --mem-per-gpu=100G --container-name=used-container --container-mounts=/shared:/shared python /shared/home/$USER/customization_simple/train.py`

**Error:**

![Screenshot 2025-04-28 at 15 54 56](https://github.com/user-attachments/assets/97cd91de-0cd5-4eea-b1c4-d75722566948)
![Screenshot 2025-04-28 at 15 55 18](https://github.com/user-attachments/assets/92f4fa28-501a-48c5-ab9e-b50ce2769090)

**Hardware:**

![Screenshot 2025-04-28 at 15 59 31](https://github.com/user-attachments/assets/2935cb9e-927d-4b7b-a12f-627070af247c)

![Screenshot 2025-04-28 at 16 00 30](https://github.com/user-attachments/assets/1473bf77-d0a1-4bfb-83da-4bb0788a766a)
