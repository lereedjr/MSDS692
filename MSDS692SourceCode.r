##
## MSDS692
###
## Download raw data file to R
## Read CSV into R
#
start_time <- Sys.time()
MushroomDLRaw <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data")
end_time <- Sys.time()
end_time - start_time

## check DownLoaded raw data
head(MushroomDLRaw)
str(MushroomDLRaw)
summary(MushroomDLRaw)

#
## Setup for RDBMS connectivity
## Connect to SQL Server
## DB is designated in OBDC setup through windows
#
library(DBI)
library(odbc)
library(RODBC)

con <- odbcConnect(odbc(),Driver,Server,UID,PWD);

## rcon <- dbConnect(odbc(),Driver,Server,UID,PWD);
rcon <- dbConnect(odbc(),
                 Driver = "SQL Server",
                 Server = "DADS_LAPTOP",
                 Database = "MSDS692",
                 UID = "XXXX",
                 PWD = "YYYY");


## -----------------------------------------------------------------------------
#
## Connection Strings
# 

# connecting to Azure SQL Server
# Driver={ODBC Driver 13 for SQL Server};Server=tcp:msds692.database.windows.net
# ,1433;Database=MSDS692;
# Uid=jkiernan@msds692;Pwd={your_password_here};
# Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;
#
#
## load data to RDBMS
# 
dbWriteTable(rcon, "MushroomRawTest", MushroomRaw, overwrite = TRUE)


# -------------------------------------------------------
## pull final dataset top 12 factors and classifier
# -------------------------------------------------------
#
QRaw <- sqlQuery(con, "select [class], [sporeprintcolor], [odor], [gillsize], [ringtype], [bruises], [population],
[gillspacing], [stalkroot], [stalksurfaceabovering], [ringnumber], [stalkshape], [stalksurfacebelowring]
from [msds692].[dbo].[mrraw]")

## dataset review
ls()
rm(QryView)
str(QRaw)
summary(QRaw)

#
## -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------
#
## Data Manipulations
#
## Random Forest setup and Analysis
#
## Creating dataset for Random Forest Analysis
## Train/Test split will be 70/30
## Cleaned Data is pulled from Storage RDBMS

ls()
rm(qry)

## Import Cleaned Dataset
qry <- sqlQuery(con, "select * from [MSDS692].[dbo].[MRRaw]")
str(qry)
summary(qry)

## Split Dataset 70/30
dftmp <- sort(sample(nrow(qry), nrow(qry)*.7))
q.trn <- qry[dftmp,]
q.tst  <- qry[-dftmp,]

str(q.trn)
#
## --------------------------------------------------------
## --------------------------------------------------------
## 
## Variable Importance using "importance Fucntion" from Random Forest package
## Load Libraries for Random Forest Analysis - Variable Importance
#
library(randomForest)
library(dplyr)
library(ggplot2)
library(ggthemes)

str(q.trn[,785])
summary(q.trn[,785])
q.trn$y

head(q.trn[,785])
tail(q.trn[,785])

## Random Forest Analysis on dataset q.trn
rfmod <- randomForest(q.trn$y ~ . -q.trn$y, data = q.trn[,1:784])
rfmod

## isolate importance from model and roundoff values
importance     <- importance(rfmod)
ImportantVars  <-  data.frame(Variables = row.names(importance), Importance = round(importance,2))

## rank ImportantVars
rankImportance <- ImportantVars %>% mutate(Rank = paste0('#',dense_rank(desc(Importance))))

## Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance)
                         , y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank)
                             , hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Mushroom Attributes- Normalized') + 
  coord_flip() 
##  theme_few() << used from library(ggthemes) but not needed here

#
## --------------------------------------------------------
## --------------------------------------------------------
#
## Normalize Dataset
#

## Create function to normalize dataset; 
## this function is adapted from examples used in many R packages

## Normalize function
fn_normalize <- function(x) { return ( (x - min(x)) / (max(x) - min(x)) ) }

## The normalize function above can be applied to every column in the dataset 
## using the lapply() function:
#

q.Norm <- as.data.frame(lapply(qry, fn_normalize))

str(q.Norm)

## Comparing Cleaned/Normalized Dataset to Cleaned only.
## using the same code from above, replacing "qry" working object with new "normalized" object q.Norm
#
dftmp <- sort(sample(nrow(q.Norm), nrow(q.Norm)*.7))
q.trn <- q.Norm[dftmp,]
q.tst  <- q.Norm[-dftmp,]

str(q.tst)

## --------------------------------------------------------
## --------------------------------------------------------

## Pull Cleaned Dataset 
## - see above for code
## QRaw dataset pulled using above code 
## dataset review
ls()
rm(QryView)
str(QRaw)
summary(QRaw)

## Split Dataset 70/30
## Create Training and Test Datasets
## - using code from above

dftmp <- sort(sample(nrow(QRaw), nrow(QRaw)*.7))
q.trn <- QRaw[dftmp,]
q.tst  <- QRaw[-dftmp,]

str(q.trn)
str(q.tst)

## --------------------------------------------------------
## --------------------------------------------------------

## ANN
## specify all variable names
## n <- names(train_)
## f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
## nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
## 

## specify all variable names
q.frm <- as.formula(paste("class ~", paste(names(q.trn[,2:13]), collapse=" + ")))
length(q.frm)

#
## Run NN 
#
library(neuralnet)

#
## hidden layer = 1, 'rprop+' and 'rprop-' refer to 
## the resilient backpropagation with and without weight backtracking

start_time <- Sys.time()
q.nn1 <- neuralnet(q.frm, data = q.trn, hidden = 1, algorithm = "rprop+", stepmax = 1e+06)
end_time <- Sys.time()
end_time - start_time

## plot models
plot(q.nn1)

#
##---------------------------------------------------------------------------
#

## ANN Model Performance
## To generate predictions on the test dataset, use the compute() as follows:
  
  q.prf1 <- compute(q.nn1, q.tst[2:13])

str(q.prf1)
  
## using the internal functions of compute results
predicted_class1 <- q.prf1$net.result

## Correlation between predicted and actual in q.tst
cor(predicted_class1, q.tst$class)

## -------------------------------------------------------------
## -------------------------------------------------------------
