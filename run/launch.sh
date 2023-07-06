sudo docker container rm rawformer_exp
sudo docker build -t rawformer ../
sudo docker run \
    --gpus all \
    --name rawformer_exp \
    --shm-size=50gb \
    -v /home/shin/exp/DB:/data \
    rawformer:latest