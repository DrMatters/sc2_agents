import tsmlstarterbot

# Load the model from the models directory. Models directory is created during training.
# Run "make" to download data and train.
tsmlstarterbot.BoostingBot(location="xgb_model.pickle", name="MyBoostingBot").play()
