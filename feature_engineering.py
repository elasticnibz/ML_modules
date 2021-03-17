# Categorical embeddings using keras
class CategoricalEmbeddings:
    def __init__(self, layers=[(50, "relu"), (15, "relu")], vecSize=None, epoch=50, batchSize=4):
        """
            Arguments:
                layers    = List([Int, Int]) NN layer sizes and activation func for training autoencoder
                            default = [(50, "relu"), (15, "relu")]
                vecSize   = [int] size of the embeddings
                            default = min(50, (inputSize+1)/2)
                epoch     = [int] number of epochs autoencoder to be trained
                batchSize = [int] size of training batches
        """
        self.epochs = epoch
        self.layers = layers
        self.vecSize = vecSize
        self.labelEncoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = Sequential()
        self.feature = None
        
    def fit(self, data, feature, featureList):
        """
            Arguments:
                feature     = [string] feature column to be vectorized
                featureList = List([strings]) target feature columns to be trained on
        """
        self.feature = feature
        inputSize = data[self.feature].unique().size

        if self.vecSize == None:
            embeddingSize = int(min(50, (inputSize+1)/2))
        else:
            embeddingSize = int(self.vecSize)
        print("Feature: {}\nInput Size: {}\nEmbedding Size: {}\n".format(self.feature, inputSize, embeddingSize))
        
        self.labelEncoder.fit(data[self.feature])
        self.scaler.fit(data[featureList].values.reshape(-1,len(featureList)))
        self.model.add(Embedding(input_dim=inputSize, output_dim=embeddingSize, input_length=1, name=self.feature+"_embedding"))
        self.model.add(Flatten())
        self.model.add(Dense(self.layers[0][0], activation=layers[0][1]))
        self.model.add(Dense(self.layers[1][0], activation=layers[1][1]))
        self.model.add(Dense(1))
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        self.model.fit(x=self.labelEncoder.transform(data[self.feature].values.reshape(-1,1)), y=self.scaler.transform(data[featureList].values.reshape(-1,len(featureList))), epochs=self.epochs, batch_size=batchSize)
        return self