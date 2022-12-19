# Machine Learning - Thyroid Nodule Ultrasound Features
# Data source: http://cimalab.unal.edu.co/applications/thyroid/
# We used the XML features, not the images for this effort

# Load libraries to use
library(class)          # k-nearest neighbors
library(kknn)           # weighted k-nearest neighbors
library(rpart)          # tree plots
library(partykit)       # tree plots
library(rpart.plot)     # tree plots as well

# Load in the data
df.rawdata = as.data.frame(read.csv(file.choose(), header=TRUE))
str(df.rawdata)

# Clean the data a bit, remove the Case ID column as unnecessary
# All the other columns were imported correctly
df.rawdata = df.rawdata[-1]

# Get a few counts for analysis, e.g. how imbalanced is the data?
# There are roughly twice as many malignant as benign in the data set
numB = nrow(df.rawdata[df.rawdata$Label=='B',])
numM = nrow(df.rawdata[df.rawdata$Label=='M',])

# Shuffle the data (several times) to ensure randomization
for (i in 1:20){
  df.rawdata= df.rawdata[sample(1:nrow(df.rawdata)), ]
}

# Show a tree plot of the ultrasound characteristics we collected
noduleTree <- rpart(Label ~ ., data=df.rawdata)
rpart.plot(noduleTree, type=0, tweak=1.3, extra=2, fallen.leaves=TRUE, uniform=TRUE)

# Divide the data set into training and testing sets for KNN clustering
ind = sample(2, nrow(df.rawdata), replace=TRUE, prob=c(0.60,0.40))
df.train = df.rawdata[ind==1,]
df.test = df.rawdata[ind==2,]

# Check the imbalances of the training and testing sets
# What percent are benign? Ideally it should be about 33%
nrow(df.train[df.train$Label=='B',]) / nrow(df.train)
nrow(df.test[df.test$Label=='B',]) / nrow(df.test)

# Train a model using knn clustering
knn.train = train.kknn(formula = Label ~ ., data=df.train, kmax=5)
summary(knn.train)

# Make predictions for the test data
predictResults = predict(knn.train, df.test[-1])

# Confusion matrix results
cmatrix = table(df.test$Label, predictResults)
cmatrix

# Define some of the terms derived from the confusion matrix
TN = cmatrix[1,1] # True negative
TP = cmatrix[2,2] # True positive
FN = cmatrix[2,1] # False negative
FP = cmatrix[1,2] # False positive
ALL = TN + TP + FN + FP # All results

# Now define the critical measures such as accuracy and F1
C.ACC = (TP + TN) / ALL # Accuracy
C.TPR = TP / (TP + FN) # True positive rate (sensitivity, recall, hit rate)
C.TNR = TN / (TN + FP) # True negative rate (specificity)
C.FPR = FP / (FP + TN) # False positive rate (fall out)
C.FNR = FN / (FN + TP) # False negative rate (miss rate)
C.NPV = TN / (TN + FN) # Negative predictive value
C.PPV = TP / (TP + FP) # Positive predictive value (precision)
C.FDR = FP / (FP + TP) # False discovery rate
C.FOR = FN / (FN + TN) # False omission rate
C.F1score = (TP + TN) / ALL # F1 score

# We need to address the issue that we might have arrived at this
# result from pure chance, so we calculate Cohen's Kappa value to be
# sure about it. The rating is...
# 0.00-0.20   - Poor  
# 0.21-0.40   - Fair   
# 0.41-0.60   - Moderate   
# 0.61-0.80   - Good   
# 0.81-1.00   - Very good 
prob.chance <- ((TP + FP)/ALL) * ((TP + FN)/ALL)
kappa <- (C.ACC - prob.chance) / (1 - prob.chance)

