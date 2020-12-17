FILE=$1

if [ $FILE == "celeba" ]; then

    # CelebA images and attribute labels
    URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
    ZIP_FILE=./Datasets/celeba.zip
    mkdir -p ./Datasets/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./Datasets/
    rm $ZIP_FILE


elif [ $FILE == 'models' ]; then

    # Models that have been trained during this project
    URL=https://www.dropbox.com/s/fgc5wnql9o7u3sd/Models.zip?dl=0
    ZIP_FILE=./Models.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE 
    rm $ZIP_FILE

else
    echo "Available arguments are celeba and models"
    exit 1
fi