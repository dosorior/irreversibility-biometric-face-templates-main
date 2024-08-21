# Train model
# Based on: icip2022_face_reconstruction

import os
import sys
import numpy
import torch
import random
import numpy as np
import cv2
import time

sys.path.append(os.getcwd())
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
length_of_embedding = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("************ NOTE: The torch device is:", device)

# =================== import Dataset ======================
from src.Dataset import MyDataset
from torch.utils.data import DataLoader

image_dir = "ffhq_align/images"
embedding_dir = "ffhq_align/embeddings_pemiu_16"
save_path = "training_files_pemiu_16"

training_dataset = MyDataset(train=True,
                             device=device,
                             image_dir=image_dir,
                             embedding_dir=embedding_dir)
testing_dataset = MyDataset(train=False,
                            device=device,
                            image_dir=image_dir,
                            embedding_dir=embedding_dir)

train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
# ========================================================


# =================== import Network =====================
from src.Network import Generator

model_Generator = Generator(length_of_embedding=length_of_embedding)
model_Generator.to(device)
# ========================================================


# =================== import Loss ========================
# ***** SSIM_Loss
from src.loss.SSIMLoss import SSIM_Loss

ssim_loss = SSIM_Loss()
ssim_loss.to(device)

# ***** ID_loss
from src.loss.FaceIDLoss import ID_Loss

ID_loss = ID_Loss(device=device)

# ***** Other losses
MSE_loss = torch.nn.MSELoss()
# ========================================================


# =================== Optimizers =========================
# ***** optimizer_Generator
optimizer_Generator = torch.optim.Adam(model_Generator.parameters(), lr=1e-3)
scheduler_Generator = torch.optim.lr_scheduler.StepLR(optimizer_Generator, step_size=10, gamma=0.5)
# ========================================================


# =================== Save models and logs ===============
os.makedirs(save_path, exist_ok=True)
os.makedirs(f"{save_path}/models", exist_ok=True)
os.makedirs(f"{save_path}/Generated_images", exist_ok=True)
os.makedirs(f"{save_path}/logs_train", exist_ok=True)


def image_save(dataset: DataLoader, filename: str, index: int):
    for i in range(dataset.size(0)):
        # if i >2:
        #    break
        os.makedirs(f"{save_path}/Generated_images/{i}", exist_ok=True)
        image = dataset[i].squeeze()
        image = (image.numpy().transpose(1, 2, 0) * 255).astype(int)
        cv2.imwrite(f"{save_path}/Generated_images/{i}/{filename}_{index}.jpg",
                    np.array([image[:, :, 2], image[:, :, 1], image[:, :, 0]]).transpose(1, 2, 0))


with open(f"{save_path}/logs_train/generator.csv", 'w') as f:
    f.write("epoch,MSE_loss_Gen,ID_loss_Gen,ssim_loss_Gen_test,total_loss_Gen\n")

with open(f"{save_path}/logs_train/log.txt", 'w') as f:
    pass

for embedding, real_image in test_dataloader:
    pass

# real_image = next(iter(test_dataloader))[1].cpu()

real_image = real_image.cpu()
image_save(real_image, "real_image", 0)
# ========================================================


# =================== Train ==============================
start_time = time.time()
num_epochs = 100
for epoch in range(num_epochs):
    # Console log
    elapsed_time = round(((time.time() - start_time)/60), 3)
    if epoch > 0:
        remaining_time = (round(((elapsed_time/epoch) * (num_epochs - epoch + 1))/60, 2))
    else:
        remaining_time = (round((elapsed_time * (num_epochs - epoch + 1)/60), 2))
    print(f'epoch: {epoch + 1}, '
          f'\t learning rate: {optimizer_Generator.param_groups[0]["lr"]},'
          f'\t elapsed: {elapsed_time} min ({round((elapsed_time/60), 2)} h),'
          f'\t remaining: {remaining_time} h')

    iteration = 0
    model_Generator.train()
    for embedding, real_image in train_dataloader:
        # ==================forward==================
        generated_image = model_Generator(embedding)
        MSE = MSE_loss(generated_image, real_image)
        ID = ID_loss(generated_image, real_image)
        ssim = ssim_loss(generated_image, real_image)
        total_loss_Generator = MSE + 0.1 * ssim + 0.005 * ID

        # ==================backward=================
        optimizer_Generator.zero_grad()
        total_loss_Generator.backward()
        optimizer_Generator.step()
        # ==================log======================
        iteration += 1
        if iteration % 200 == 0:
            # model_Generator.eval()
            # print(f'epoch:{epoch+1} \t, iteration: {iteration}, \t total_loss:{total_loss_Generato.data.item()}')
            with open(f"{save_path}/logs_train/log.txt", 'a') as f:
                f.write(
                    f'epoch:{epoch + 1}, \t iteration: {iteration}, \t total_loss:{total_loss_Generator.data.item()}\n')
            pass

    # ******************** Eval Generator ********************
    model_Generator.eval()
    MSE_loss_Gen_test = ID_loss_Gen_test = ssim_loss_Gen_test = total_loss_Gen_test = 0
    iteration = 1
    for embedding, real_image in test_dataloader:
        iteration += 1
        # ==================forward==================
        with torch.no_grad():
            generated_image = model_Generator(embedding)
            MSE = MSE_loss(generated_image, real_image)
            ID = ID_loss(generated_image, real_image)
            ssim = ssim_loss(generated_image, real_image)
            total_loss_Generator = MSE + 0.1 * ssim + 0.005 * ID
            ####
            MSE_loss_Gen_test += MSE.item()
            ID_loss_Gen_test += ID.item()
            ssim_loss_Gen_test += ssim.item()
            total_loss_Gen_test += total_loss_Generator.item()

    with open(f"{save_path}/logs_train/generator.csv", 'a') as f:
        f.write(
            f"{epoch + 1}, {MSE_loss_Gen_test / iteration}, {ID_loss_Gen_test / iteration}, "
            f"{ssim_loss_Gen_test / iteration}, {total_loss_Gen_test / iteration}\n")

    generated_image = model_Generator(embedding).detach().cpu()
    image_save(generated_image, "epoch", epoch + 1)
    # *******************************************************

    # Save model_Generator
    torch.save(model_Generator.state_dict(), f"{save_path}/models/Generator_{epoch + 1}.pth")

    # Update optimizer_Generator lr
    scheduler_Generator.step()
# ========================================================
