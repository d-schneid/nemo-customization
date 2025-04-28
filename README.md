I try with efa-enabled container: nemo_efa_2502

**Command does not work with** `pipeline_model_parallel_size=2`:

`srun --ntasks-per-node=8 --gres=gpu:8 --mem-per-gpu=100G --container-name=nemo_efa_2502 --container-mounts=/shared:/shared python /shared/home/$USER/customization_simple/train.py`

**It gets stuck here forever:**

![image](https://github.com/user-attachments/assets/8c2c21d4-8dd7-4274-bcd3-a122ca24ef7a)

**Hardware:**

![Screenshot 2025-04-28 at 15 59 31](https://github.com/user-attachments/assets/2935cb9e-927d-4b7b-a12f-627070af247c)

![Screenshot 2025-04-28 at 16 00 30](https://github.com/user-attachments/assets/1473bf77-d0a1-4bfb-83da-4bb0788a766a)
