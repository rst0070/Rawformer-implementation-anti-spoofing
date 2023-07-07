FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update

RUN pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pip --upgrade
RUN pip install wandb
RUN pip install torch_audiomentations

COPY ./ /app/
WORKDIR /app

CMD ["python",  "main.py"]