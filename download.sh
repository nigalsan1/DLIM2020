URL=https://www.dropbox.com/s/h8h2mebvugsppzn/test2.zip?dl=0
ZIP_FILE=./Models/Models.zip
mkdir -p ./Models/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./Models/
rm $ZIP_FILE

URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
ZIP_FILE=./Datasets/celeba.zip
mkdir -p ./Datasets/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./StarGAN/data/
rm $ZIP_FILE