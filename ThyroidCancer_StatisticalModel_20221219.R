# Statistical model for NOVA Science Publishers Chapter:
# "Statistics of Thyroid Cancer: 
# Overdiagnosis and Machine Learning Approaches"
# Matthew S Neel, 2023

# Here we simulate the lives of a group of people and their 
# experience (or lack of it) with thyroid nodule appearance, 
# growth, and effect on their life span in the United States


# ------------------------
# ---- LOAD LIBRARIES ----
# ------------------------


# Load libraries that we'll need
library(fGarch)
library(tidyverse)
library(dplyr)
library(memisc)
library(matrixStats)
library(stringr)
library(class)
library(kknn)

# Set the random number seed
set.seed(314)

# -------------------------------
# ---- CREATE THE POPULATION ----
# -------------------------------


# First we need distribution functions for the normal life 
# expectancy of men and women in the US. Following the distribution
# presented by Edwards, 2008, Cost of Uncertain Lifespan, Fig1
# This distribution was a negative skew, normal distribution
# We assume our subjects are 50% male / 50% female
N <- 10000  # Number of subjects in our simulation
lifeExpectMen = as.integer(rsnorm(N/2, mean = 65, sd = 15, xi = -1.5))
lifeExpectWomen = as.integer(rsnorm(N/2, mean = 70, sd = 15, xi = -1.5))

# Now let's make a data frame that stores our subjects gender 
# and life expectancy
df.men = data.frame('LifeExpect' = lifeExpectMen, 'Sex' = 'M')
df.women = data.frame('LifeExpect' = lifeExpectWomen, 'Sex' = 'F')


# -----------------------------------------------
# ---- NODULE INCIDENCE: HOW/WHEN THEY OCCUR ----
# -----------------------------------------------


# Now for the big question: When will nodules appear in a person's life?
# Thyrocytes have a notoriously slow cell regeneration rate of about
# five times in a person's life. See Ozaki (2012)
# Thus, we will flip a weighted coin five times at different ages in a person's life
# Probability of thyroid nodules increases with age. See Kamran (2013)
# Nodules are four times more common in women than men; see Welker (2003)
df.men$TN1 = sample(0:1, size=N/2, prob=c(0.95,0.05), rep=T)
df.men$TN2 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)
df.men$TN3 = sample(0:1, size=N/2, prob=c(0.85,0.15), rep=T)
df.men$TN4 = sample(0:1, size=N/2, prob=c(0.80,0.20), rep=T)
df.men$TN5 = sample(0:1, size=N/2, prob=c(0.75,0.25), rep=T)
df.women$TN1 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)
df.women$TN2 = sample(0:1, size=N/2, prob=c(0.75,0.25), rep=T)
df.women$TN3 = sample(0:1, size=N/2, prob=c(0.50,0.50), rep=T)
df.women$TN4 = sample(0:1, size=N/2, prob=c(0.25,0.75), rep=T)
df.women$TN5 = sample(0:1, size=N/2, prob=c(0.10,0.90), rep=T)

# Now we'll need a weighted coin for the malignant / benign 
# state of a thyroid nodule. We follow Bomeli 2010 when it states
# 5-10% of thyroid nodules are malignant (p1, Introduction). 
# Here we'll use 7% on average for the coin flip
# But malignancy is about 4:1 for PTC / FTC and 1:1 for medullary / anaplastic 
# for men as women (Le Clair, 2021))
# For men we'll use 4% for men and 10% for women
df.men$BM1 = sample(0:1, size=N/2, prob=c(0.96,0.04), rep=T)
df.men$BM2 = sample(0:1, size=N/2, prob=c(0.96,0.04), rep=T)
df.men$BM3 = sample(0:1, size=N/2, prob=c(0.96,0.04), rep=T)
df.men$BM4 = sample(0:1, size=N/2, prob=c(0.96,0.04), rep=T)
df.men$BM5 = sample(0:1, size=N/2, prob=c(0.96,0.04), rep=T)
df.women$BM1 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)
df.women$BM2 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)
df.women$BM3 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)
df.women$BM4 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)
df.women$BM5 = sample(0:1, size=N/2, prob=c(0.90,0.10), rep=T)

# Now that we have performed all the gender-influenced coin flips, let's
# combine the two data frames into a single one and shuffle the records
df.subjects = union_all(df.men, df.women)

# We also need a coin that tells us what type of cancer the malignant
# node is. There are four types with particular probabilities.
# Anaplastic = 2% of cases
# Medullary = 4%
# Follicular = 14% 
# Papillary = 80%
# Here we are following the percentages given in...
# https://www.cancer.org/cancer/thyroid-cancer/about/what-is-thyroid-cancer.html
df.subjects$MType1 = sample(c('A','M','F','P'), size=N, prob=c(0.02,0.04,0.14,0.80), rep=T)
df.subjects$MType2 = sample(c('A','M','F','P'), size=N, prob=c(0.02,0.04,0.14,0.80), rep=T)
df.subjects$MType3 = sample(c('A','M','F','P'), size=N, prob=c(0.02,0.04,0.14,0.80), rep=T)
df.subjects$MType4 = sample(c('A','M','F','P'), size=N, prob=c(0.02,0.04,0.14,0.80), rep=T)
df.subjects$MType5 = sample(c('A','M','F','P'), size=N, prob=c(0.02,0.04,0.14,0.80), rep=T)

# We also need a coin that determines the size of the nodule, if present
# This has a direct bearing on detection via ultrasound / palpable / etc.
# The value is diameter of the spherical nodule in mm
# This should follow an inverse curve, i.e. the probability of the nodule
# having that size is the inverse of the size. In other words, smaller
# nodules are much more common (and more often ignored) than larger ones
# See Ha (2021) and Guth (2009) and Ross (2002)
df.subjects$Size1 = ceiling(10*rexp(N, rate = 3))
df.subjects$Size2 = ceiling(10*rexp(N, rate = 3))
df.subjects$Size3 = ceiling(10*rexp(N, rate = 3))
df.subjects$Size4 = ceiling(10*rexp(N, rate = 3))
df.subjects$Size5 = ceiling(10*rexp(N, rate = 3))

# Now we need to find the ages at which the TN coin is flipped
# Everyone gets a coin flip in one of the five age brackets
# Age brackets are 15-30,30-45,45-60,60-75,75-100
# We shape the probability for the first flip since early-age nodules are very rate
p.weight = c(1:15)/120
df.subjects$Flip1Age = sample(c(15:29), prob=p.weight, size=N, rep=T)
df.subjects$Flip2Age = sample(c(30:44), size=N, rep=T)
df.subjects$Flip3Age = sample(c(45:59), size=N, rep=T)
df.subjects$Flip4Age = sample(c(60:74), size=N, rep=T)
df.subjects$Flip5Age = sample(c(75:89), size=N, rep=T)

# Now we need to find the age at death assuming that a person will always reach
# their life expectancy unless a malignant nodule causes an early death
# We assume that there is NO medical treatment available to anyone to treat the cancer
# In reality this would be a complex function of age, cancer type, nodule size, etc.
# See Davies 2010 as a start
# We will simply assign a number to the decision that represents how much longer
# that subject will live. For example, if we had...
# df.subjects$horizon30 = 30 + 7
# ... then the person would live to age 37
# If a person has two horizons, then we choose the earlier of the two, etc.
# Papillary  = horizon of 25 years ... barely classifiable as cancer, slow-moving
# Follicular = horizon of 20 years
# Medullary  = horizon of 10 years
# Anaplastic = horizon of  2 years ... aggressive, fast-killing cancer
hPap = 25
hFol = 20
hMed = 10
hAna = 2

# Now we setup the life horizon for each age level
# Start with the default of life expexctancy
df.subjects$HZN1 = 1000 # df.subjects$LifeExpect
df.subjects$HZN2 = 1000 # df.subjects$LifeExpect
df.subjects$HZN3 = 1000 # df.subjects$LifeExpect
df.subjects$HZN4 = 1000 # df.subjects$LifeExpect
df.subjects$HZN5 = 1000 # df.subjects$LifeExpect

# Handle the age 1 set
df.subjects$HZN1[df.subjects$TN1==1 & df.subjects$BM1==1 & df.subjects$MType1=='P'] = hPap
df.subjects$HZN1[df.subjects$TN1==1 & df.subjects$BM1==1 & df.subjects$MType1=='F'] = hFol
df.subjects$HZN1[df.subjects$TN1==1 & df.subjects$BM1==1 & df.subjects$MType1=='M'] = hMed
df.subjects$HZN1[df.subjects$TN1==1 & df.subjects$BM1==1 & df.subjects$MType1=='A'] = hAna

# Handle the age 2 set
df.subjects$HZN2[df.subjects$TN2==1 & df.subjects$BM2==1 & df.subjects$MType2=='P'] = hPap
df.subjects$HZN2[df.subjects$TN2==1 & df.subjects$BM2==1 & df.subjects$MType2=='F'] = hFol
df.subjects$HZN2[df.subjects$TN2==1 & df.subjects$BM2==1 & df.subjects$MType2=='M'] = hMed
df.subjects$HZN2[df.subjects$TN2==1 & df.subjects$BM2==1 & df.subjects$MType2=='A'] = hAna

# Handle the age 3 set
df.subjects$HZN3[df.subjects$TN3==1 & df.subjects$BM3==1 & df.subjects$MType3=='P'] = hPap
df.subjects$HZN3[df.subjects$TN3==1 & df.subjects$BM3==1 & df.subjects$MType3=='F'] = hFol
df.subjects$HZN3[df.subjects$TN3==1 & df.subjects$BM3==1 & df.subjects$MType3=='M'] = hMed
df.subjects$HZN3[df.subjects$TN3==1 & df.subjects$BM3==1 & df.subjects$MType3=='A'] = hAna

# Handle the age 4 set
df.subjects$HZN4[df.subjects$TN4==1 & df.subjects$BM4==1 & df.subjects$MType4=='P'] = hPap
df.subjects$HZN4[df.subjects$TN4==1 & df.subjects$BM4==1 & df.subjects$MType4=='F'] = hFol
df.subjects$HZN4[df.subjects$TN4==1 & df.subjects$BM4==1 & df.subjects$MType4=='M'] = hMed
df.subjects$HZN4[df.subjects$TN4==1 & df.subjects$BM4==1 & df.subjects$MType4=='A'] = hAna

# Handle the age 5 set
df.subjects$HZN5[df.subjects$TN5==1 & df.subjects$BM5==1 & df.subjects$MType5=='P'] = hPap
df.subjects$HZN5[df.subjects$TN5==1 & df.subjects$BM5==1 & df.subjects$MType5=='F'] = hFol
df.subjects$HZN5[df.subjects$TN5==1 & df.subjects$BM5==1 & df.subjects$MType5=='M'] = hMed
df.subjects$HZN5[df.subjects$TN5==1 & df.subjects$BM5==1 & df.subjects$MType5=='A'] = hAna

# Finally, we calculate the age at death, which might be from living out the full life
# expectancy or having their life cut short by cancer
# We look across all the horizons and take the minimum
# We include the life expectancy in the group so that no one's life is mathematically
# "extended" by a coin flip giving +20 years beyond the expectancy
df.subjects$AgeAtDeath = with(df.subjects, pmin(
  LifeExpect,
  HZN1+Flip1Age,
  HZN2+Flip2Age,
  HZN3+Flip3Age,
  HZN4+Flip4Age,
  HZN5+Flip5Age))

# Define a column that lets us easily identify those who died of thyroid cancer
df.subjects$DeathCause = 'Other'
df.subjects$DeathCause[df.subjects$AgeAtDeath<df.subjects$LifeExpect] = 'Cancer'

# Define a column that counts how many nodules are malignant, if any
df.subjects$Mcount = rowSums(df.subjects[,8:12])

# We need to screen out the coin flips that couldn't have happened due to the
# subject dying before the appointed time points
# For example, if the subject dies at age 53, the coin flip for age 60 must be 0
# We assume, by necessity, that all subjects live to at least 15 years old
# So there is no need to handle TN15 as we do the other coin flips below
df.subjects$TN1[df.subjects$AgeAtDeath<df.subjects$Flip1Age] = 0
df.subjects$TN2[df.subjects$AgeAtDeath<df.subjects$Flip2Age] = 0
df.subjects$TN3[df.subjects$AgeAtDeath<df.subjects$Flip3Age] = 0
df.subjects$TN4[df.subjects$AgeAtDeath<df.subjects$Flip4Age] = 0
df.subjects$TN5[df.subjects$AgeAtDeath<df.subjects$Flip5Age] = 0

# Let's eliminate values that make no sense when there isn't a thyroid nodule
df.subjects$BM1[df.subjects$TN1==0] = NA
df.subjects$BM2[df.subjects$TN2==0] = NA
df.subjects$BM3[df.subjects$TN3==0] = NA
df.subjects$BM4[df.subjects$TN4==0] = NA
df.subjects$BM5[df.subjects$TN5==0] = NA

df.subjects$MType1[df.subjects$TN1==0] = NA
df.subjects$MType2[df.subjects$TN2==0] = NA
df.subjects$MType3[df.subjects$TN3==0] = NA
df.subjects$MType4[df.subjects$TN4==0] = NA
df.subjects$MType5[df.subjects$TN5==0] = NA

df.subjects$Size1[df.subjects$TN1==0] = 0
df.subjects$Size2[df.subjects$TN2==0] = 0
df.subjects$Size3[df.subjects$TN3==0] = 0
df.subjects$Size4[df.subjects$TN4==0] = 0
df.subjects$Size5[df.subjects$TN5==0] = 0

# Here we'll add some more columns and begin to normalize the data into
# subordinate data frames for querying

# Define a column that counts total thyroid nodules for each subject
df.subjects$TNtotal = rowSums(df.subjects[,3:7])

# Find the size of the largest nodule for that subject
df.subjects$MaxSize = rowMaxs(as.matrix(df.subjects[,c(18:22)]))

# Assign a Subject ID to each person
df.subjects$SubjectID = as.character(c(10000:19999))

# Assign an ID to a specific thyroid nodule
df.reps = data.frame('ID1' = replicate(10000,'1'),
                     'ID2' = replicate(10000,'2'),
                     'ID3' = replicate(10000,'3'),
                     'ID4' = replicate(10000,'4'),
                     'ID5' = replicate(10000,'5')
                     )
df.subjects$TN1ID = paste(df.subjects$SubjectID, df.reps$ID1, sep="-")
df.subjects$TN2ID = paste(df.subjects$SubjectID, df.reps$ID2, sep="-")
df.subjects$TN3ID = paste(df.subjects$SubjectID, df.reps$ID3, sep="-")
df.subjects$TN4ID = paste(df.subjects$SubjectID, df.reps$ID4, sep="-")
df.subjects$TN5ID = paste(df.subjects$SubjectID, df.reps$ID5, sep="-")

# Make a new DF to describe the people at a high level
df.people = df.subjects[c(1,2,33:38)]

# Create sub tables for each age event
df.N1 = df.subjects[c(3,8,13,18,23,28,33,38,39,2,1)]
df.N2 = df.subjects[c(4,9,14,19,24,29,33,38,40,2,1)]
df.N3 = df.subjects[c(5,10,15,20,25,30,33,38,41,2,1)]
df.N4 = df.subjects[c(6,11,16,21,26,31,33,38,42,2,1)]
df.N5 = df.subjects[c(7,12,17,22,27,32,33,38,43,2,1)]

# Rename the columns so they agree
colnames(df.N1) = c('Presence','BM','Type','Size','FlipAge','Horizon','AgeAtDeath','SubjectID','NoduleID','Sex','LifeExpect')
colnames(df.N2) = c('Presence','BM','Type','Size','FlipAge','Horizon','AgeAtDeath','SubjectID','NoduleID','Sex','LifeExpect')
colnames(df.N3) = c('Presence','BM','Type','Size','FlipAge','Horizon','AgeAtDeath','SubjectID','NoduleID','Sex','LifeExpect')
colnames(df.N4) = c('Presence','BM','Type','Size','FlipAge','Horizon','AgeAtDeath','SubjectID','NoduleID','Sex','LifeExpect')
colnames(df.N5) = c('Presence','BM','Type','Size','FlipAge','Horizon','AgeAtDeath','SubjectID','NoduleID','Sex','LifeExpect')

# Now union those together to build a nodule database
df.nodules = rbind(df.N1,df.N2,df.N3,df.N4,df.N5)

# Redesign the horizon for malignant nodules 
df.nodules$Horizon[df.nodules$Type=='P'] = hPap
df.nodules$Horizon[df.nodules$Type=='F'] = hFol
df.nodules$Horizon[df.nodules$Type=='M'] = hMed
df.nodules$Horizon[df.nodules$Type=='A'] = hAna

# Determine if this nodule was the one that killed the person or not
# Need to include the last condition below since flip age + horizon might
# equal the natural life expectancy. We consider those people NOT to have
# died from cancer since their life ended for that or other reasons unknown
df.nodules$Fatal = 'N'
df.nodules$mathCheck = df.nodules$Horizon + df.nodules$FlipAge
df.nodules$Fatal[df.nodules$mathCheck==df.nodules$AgeAtDeath 
                 & df.nodules$BM==1
                 & df.nodules$AgeAtDeath<df.nodules$LifeExpect] = 'Y'

# Need to clean the model further; remove all the non-events
df.nodules = df.nodules[df.nodules$Presence==1,]

# Remove the mathCheck column, no longer needed
df.nodules = df.nodules[-c(13)]

# Set some values to NA since they aren't applicable
df.nodules$Horizon[df.nodules$BM==0] = NA
df.nodules$Type[df.nodules$BM==0] = NA

# Now, remove the Presence column since it's no longer needed
df.nodules = df.nodules[-c(1)]

# Create a "palpable" flag for nodules: larger than 10mm qualifies
df.nodules$Palpable = 'N'
df.nodules$Palpable[df.nodules$Size>10] = 'Y'


# ---------------------------------------------
# ---- NODULE PREVALENCE: TREND OVER YEARS ----
# ---------------------------------------------


# Now we want to see the data on how many nodules / patients with nodules
# we have each year over the course of about 100 years
df.trends = data.frame('Year'=seq(1:150))
df.trends$Bcount = 0
df.trends$Mcount = 0
df.trends$NodCount = 0
df.trends$Patients = 0
df.trends$Men = 0
df.trends$Women = 0

# Start a loop through the nodules (must be done this way)
# Find a count of all nodules, NodCount = Bcount + Mcount
# Also, find a unique count of people (M/F) each year having at least one nodule
for (i in 1:150){
  df.trends$NodCount[i] = nrow(df.nodules[df.nodules$FlipAge<=df.trends$Year[i] 
                               & df.nodules$AgeAtDeath>=df.trends$Year[i],])
  df.trends$Bcount[i] = nrow(df.nodules[df.nodules$FlipAge<=df.trends$Year[i] 
                               & df.nodules$AgeAtDeath>=df.trends$Year[i]
                               & df.nodules$BM==0,])
  df.trends$Mcount[i] = nrow(df.nodules[df.nodules$FlipAge<=df.trends$Year[i] 
                               & df.nodules$AgeAtDeath>=df.trends$Year[i]
                               & df.nodules$BM==1,]) 
  df.trends$Patients[i] = n_distinct(df.nodules$SubjectID[df.nodules$FlipAge<=df.trends$Year[i] 
                               & df.nodules$AgeAtDeath>=df.trends$Year[i]])
  df.trends$Men[i] = n_distinct(df.nodules$SubjectID[df.nodules$FlipAge<=df.trends$Year[i] 
                                & df.nodules$AgeAtDeath>=df.trends$Year[i]
                                & df.nodules$Sex=='M'])
  df.trends$Women[i] = n_distinct(df.nodules$SubjectID[df.nodules$FlipAge<=df.trends$Year[i] 
                                & df.nodules$AgeAtDeath>=df.trends$Year[i]
                                & df.nodules$Sex=='F'])
  df.trends$M_Men[i] = n_distinct(df.nodules$SubjectID[df.nodules$FlipAge<=df.trends$Year[i] 
                                                     & df.nodules$AgeAtDeath>=df.trends$Year[i]
                                                     & df.nodules$Sex=='M'
                                                     & df.nodules$BM==1])
  df.trends$M_Women[i] = n_distinct(df.nodules$SubjectID[df.nodules$FlipAge<=df.trends$Year[i] 
                                                       & df.nodules$AgeAtDeath>=df.trends$Year[i]
                                                       & df.nodules$Sex=='F'
                                                       & df.nodules$BM==1])
}


# --------------------------------------
# ---- NODULE DETECTION: ULTRASOUND ----
# --------------------------------------


# Our detection of nodules in a given patient depends upon several things...
# "About 80% of people had a physician visit in the last 12 months..." (CDC.gov, 2021, 
# interactive summary stats for adults, https://wwwn.cdc.gov/NHISDataQueryTool/SHS_adult/index.html)
# "Women are 33% more likely to visit a physician than men..." (Wang, 2013)

# We assume that ultrasound has a resolution of 1mm, such that any object >=1mm
# is 100% certain to be detected

# "One ultrasound scan is performed for every three Canadians each year, 
# of which three to five per cent are thyroid ultrasounds..." (U of Alberta)
# (https://www.ualberta.ca/folio/2021/05/ai-innovation-will-make-thyroid-ultrasounds-faster-and-easier.html)
# We assume similar frequency for US citizens, or about 2% of people receive
# neck scans each year for various reasons that would lead to the discovery of
# thyroid nodules (incidental or intentional)

# Our approach is the following...
# Each year, 80% are randomly selected for an office visit (95% F / 65% M)
#   For those selected, 2% will be randomly selected for a neck scan (which will find
#     all nodules) and the remaining 98% will not have a neck scan and only palpable
#     nodules will be found

# Gather these groups separately
df.womenD = df.people[df.people$Sex=='F',]
df.menD = df.people[df.people$Sex=='M',]

# Let's create a detected flag and year found for nodules data frame
df.nodules$Detected = 'N'
df.nodules$YearFound = 1000

# Creation a new data frame to store detection results
# It will store the year that a nodule is detected with a column
# for each year we scan through, so e.g. 100 columns
df.detect = data.frame('NoduleID' = df.nodules$NoduleID)
df.detect$YearFound = 1000
df.detect$NextYear = 0

# Step into the annual loop now...

# For loop that goes from year = 15 to year = 110
for (currentYear in 15:120){

    # Randomly sample to find those who will visit the doctor this year (Phase 1)
    # Note that not all of these people will even have any nodules!
    # These people will have their palpable nodules detected but NOT the impalpable ones
    df.phase1.women = df.womenD[sample(nrow(df.womenD),0.95*(N/2)),]
    df.phase1.men = df.menD[sample(nrow(df.menD),0.65*(N/2)),]
    df.phase1 = union_all(df.phase1.men, df.phase1.women)
    
    # Now, of those, randomly select 2% for a neck scan (Phase 2)
    # Note that not all of these people will even have any nodules!
    # These people will ALSO have all their impalpable nodules detected
    df.phase2.women = df.phase1.women[sample(nrow(df.phase1.women),0.02*nrow(df.phase1.women)),]
    df.phase2.men = df.phase1.men[sample(nrow(df.phase1.men),0.02*nrow(df.phase1.men)),]
    df.phase2 = df.phase1[sample(nrow(df.phase1),0.02*nrow(df.phase1)),]

    # Creating left joins so we can match phase 1 and phase 2 subjects to 
    # the nodules they may or may not have
    # Let's start with Phase 1 information
    df.joined1 = left_join(df.nodules, df.phase1, by = "SubjectID", copy=FALSE, keep=FALSE)
    
    # Set the detection flag for all those we found; also set the year we found it
    df.joined1$Detected[!is.na(df.joined1$LifeExpect.y) 
                       & df.joined1$Palpable=='Y' 
                       & df.joined1$FlipAge<=currentYear ] = 'Y'
    df.joined1$YearFound[df.joined1$Detected=='Y'] = currentYear
    
    # Provide new data to this column; find the min; reset the data column
    df.detect$NextYear = df.joined1$YearFound
    df.detect$YearFound = with(df.detect, pmin(YearFound,NextYear))
    df.detect$NextYear = 0

    # Now move on to Phase 2
    df.joined2 = left_join(df.nodules, df.phase2, by = "SubjectID", copy=FALSE, keep=FALSE)
    
    # Set the detection flag for all those we found; also set the year we found it
    df.joined2$Detected[!is.na(df.joined2$LifeExpect.y) 
                        & df.joined2$Palpable=='N' 
                        & df.joined2$FlipAge<=currentYear ] = 'Y'
    df.joined2$YearFound[df.joined2$Detected=='Y'] = currentYear
    
    # Provide new data to this column; find the min; reset the data column
    df.detect$NextYear = df.joined2$YearFound
    df.detect$YearFound = with(df.detect, pmin(YearFound,NextYear))
    df.detect$NextYear = 0

    # Now we end the loop and return back to do it again.

}

# Report the number of nodules we found
nrow(df.detect[df.detect$YearFound<1000,])

# Bring the data back to the main nodules data frame
df.nodules$YearFound = df.detect$YearFound

# Set the detection flag where appropriate
df.nodules$Detected[df.nodules$YearFound<1000] = 'Y'

# Set the year found to NA where appropriate
df.nodules$YearFound[df.nodules$Detected=='N'] = NA

# We need to reset the YearFound that occurred after the subject died
# We also need to set the flag back to N if that occurred
df.nodules$YearFound[df.nodules$YearFound>df.nodules$AgeAtDeath] = NA
df.nodules$Detected[is.na(df.nodules$YearFound)] = 'N'

# Check the number of nodules detected now
nrow(df.nodules[df.nodules$Detected=='Y',])


# -------------------------------------
# ---- NODULE PREDICTION: FNA & US ----
# -------------------------------------


# Once a nodule is detected and is greater than 1cm in diameter, we assume the patient
# is sent to have a Fine Needle Aspiration Biopsy (FNAB) to determine the
# nature of the nodule: benign, malignant, can't tell
# In general, FNAB is only performed on nodules > 1cm (Pinchot, 2009)
# In general, FNAB returns a non-result about 25% of the time (Amrikachi, 2001)
# When FNA does return a result, it is usually with accuracy > 90% (Amrikachi, 2001)

# For those patients with nodules >=10mm (palpable in our definition) and that were
# detected by ultrasound, we flip a weighted coin to see if the FNA actually returns
# a result (75%) or not (25%)
df.nodules$FNAstatus = NA
eligibleNodules = nrow(df.nodules[df.nodules$Palpable=='Y' & df.nodules$Detected=='Y',])
df.nodules$FNAstatus[df.nodules$Palpable=='Y' & df.nodules$Detected=='Y'] = sample(c('Y','N'), size=eligibleNodules, prob=c(0.75,0.25), rep=T)

# For those that meet the criteria and get a definite FNA result, we assume 
# that FNA has 90% accuracy. We flip a coin and 90% of the time we take the 
# true value and 10% of the time we take the opposite value
df.nodules$NotBM = df.nodules$BM + 1
df.nodules$NotBM[df.nodules$NotBM==2] = 0
df.nodules$FNAresult = NA
df.nodules$BMchoice <- apply(df.nodules, 1, function(x) sample(c(x[1], x[16]), prob=c(0.90,0.10), 1))
df.nodules$FNAresult[!is.na(df.nodules$FNAstatus)] = df.nodules$BMchoice[!is.na(df.nodules$FNAstatus)]
df.nodules$FNAresult[df.nodules$FNAstatus=='N'] = NA

# But what about nodules that were detected but not palpable / large enough?
# We assume that these will be evaluated by a radiologist, whose typical accuracy is
# about 70% (Wu, 2021) when they do make a call for B/M. We assume that same 
# frequency of returning a result (75%) or not (25%) as FNA statistics.
# Following the same process as before...
df.nodules$USstatus = NA
eligibleNodules = nrow(df.nodules[df.nodules$Palpable=='N' & df.nodules$Detected=='Y',])
df.nodules$USstatus[df.nodules$Palpable=='N' & df.nodules$Detected=='Y'] = sample(c('Y','N'), size=eligibleNodules, prob=c(0.75,0.25), rep=T)
df.nodules$USresult = NA
df.nodules$BMchoice <- apply(df.nodules, 1, function(x) sample(c(x[1], x[16]), prob=c(0.70,0.30), 1))
df.nodules$USresult[!is.na(df.nodules$USstatus)] = df.nodules$BMchoice[!is.na(df.nodules$USstatus)]
df.nodules$USresult[df.nodules$USstatus=='N'] = NA


# ---------------------------------------
# ---- NODULE PREDICTION: VALIDATION ----
# ---------------------------------------


# Now that we have predictions for some of the nodules, let's see if they were correct
# First setup the field
df.nodules$Predict = NA

# Now we address the correct cases: two methods with two outcomes
df.nodules$Predict[(df.nodules$FNAresult==0 & df.nodules$BM==0)] = 'Y'
df.nodules$Predict[(df.nodules$FNAresult==1 & df.nodules$BM==1)] = 'Y'
df.nodules$Predict[(df.nodules$USresult==0 & df.nodules$BM==0)] = 'Y'
df.nodules$Predict[(df.nodules$USresult==1 & df.nodules$BM==1)] = 'Y'

# Now we address the incorrect cases...
df.nodules$Predict[(df.nodules$FNAresult==0 & df.nodules$BM==1)] = 'N'
df.nodules$Predict[(df.nodules$FNAresult==1 & df.nodules$BM==0)] = 'N'
df.nodules$Predict[(df.nodules$USresult==0 & df.nodules$BM==1)] = 'N'
df.nodules$Predict[(df.nodules$USresult==1 & df.nodules$BM==0)] = 'N'


# -----------------------------------------
# ---- NODULE PREDICTION: ML ALGORITHM ----
# -----------------------------------------


# Create a data frame that will hold the features and label
df.total = df.nodules[c(1,3,4,9)]
df.total$BM = as.factor(df.total$BM)
df.total$Sex = as.factor(df.total$Sex)

# Shuffle the rows to randomize the data a bit
for (i in 1:20){
  df.total= df.total[sample(1:nrow(df.total)), ]
}

# Create the training / testing data sets
df.training = df.total[1:7000,]
df.test = df.total[7001:nrow(df.total),]

# Train a model for prediction
model.glm <- glm(BM ~ FlipAge + Sex + Size, data=df.training, family = 'binomial')
summary(model.glm)

# Now use the model to make predictions
model.predict<-predict(model.glm, df.test, type = 'response')

# Analyze the results to see how they are grouped and find a mid-point
summary(model.predict)
hist(model.predict)
model.predict

# Post-process the results back into binary values of 1/0
model.predict<-ifelse(model.predict > 0.065, 1, 0)
model.predict

# Create the confusion matrix using predictions and actual values
table(Prediction=model.predict, Actual=df.test$BM)


# ----------------------------------------
# ---- CHECK DATA SYNC BETWEEN TABLES ----
# ----------------------------------------


# Need to check the data between the three tables: subjects, people and nodules

# Count of people who never had a nodule, these two should match
nrow(df.people[df.people$TNtotal==0,])
nrow(df.subjects[df.subjects$TNtotal==0,])

# Count of all nodules, these two should match
sum(df.subjects$TNtotal)
nrow(df.nodules)

# Count the number of people who had >0 nodules, these two should match
n_distinct(df.nodules$SubjectID)
nrow(df.people[df.people$TNtotal>0,])


# --------------------------
# ---- ANALYZE THE DATA ----
# --------------------------


# Now that we have a data set, we can ask various questions to tune the model
# and ensure that our results are at least approximately consistent with the
# research literature

# Count who died before their full life, should be a small set
nrow(df.subjects[df.subjects$DeathCause=='Cancer',])

nrow(df.subjects[df.subjects$DeathCause=='Cancer' & df.subjects$Sex=='F',])

# Count whose lives were mathematically extended by cancer, should be zero records
nrow(df.subjects[df.subjects$AgeAtDeath>df.subjects$LifeExpect,])

# Count who lived their full expectancy
nrow(df.subjects[df.subjects$AgeAtDeath==df.subjects$LifeExpect,])

# How many subjects died with no thyroid nodules at all (of any size)?
# Some studies report that >40% of autopsies find at least 1 nodule (Menderico, 2021)
# This is probably 40-60% as a value then
nrow(df.subjects[df.subjects$TNtotal==0,])

# This is a table of the ages at death for those who died of cancer
table(df.subjects$AgeAtDeath[df.subjects$DeathCause=='Cancer'])

# Create a frequency table for nodule size and number of occurrences
table(df.nodules$Size)

# Create a frequency table for palpable vs malignant
# "...about 5 percent of all palpable nodules are found to be malignant" (Welker 2003)
table(df.nodules$Palpable, df.nodules$BM)

# Assuming that a palpable thyroid nodule is at least 10mm in diameter
# how many subjects had at least one of those?
# Mayo Clinic website: "palpable nodules found in 4-7% of the adult US population"
n_distinct(df.nodules$SubjectID[df.nodules$Palpable=='Y'])

# Create tables of frequency of nodules by sex for comparison
# This is sometimes quoted at "women are four times more likely to have 
# thyroid nodules than men" (See e.g. Welker 2003)
table(df.subjects$TNtotal,df.subjects$Sex)

# Count of females and males with at least one malignant nodule
# Should be between 4:1 and 1:1 (F:M), perhaps 2:1 (Le Clair, 2021)
nrow(df.subjects[df.subjects$Mcount>0&df.subjects$Sex=="F",])
nrow(df.subjects[df.subjects$Mcount>0&df.subjects$Sex=="M",])

# On average, how many nodules per people who have them?
# This is a rough measure of multi-nodularity
nrow(df.nodules) / n_distinct(df.nodules$SubjectID)

# On average, how many nodules per person in the entire population?
# This includes people with and without any nodules, probably less than 1
nrow(df.nodules) / N

# Create a simple plot of prevalence or number of nodules/patients present in each year
plot(df.trends$Year,df.trends$Patients, col="red")
nrow(df.nodules[df.nodules$Detected=='Y',])
nrow(df.nodules[df.nodules$Detected=='N' & df.nodules$Palpable=='Y' & df.nodules$BM==1,])

# Create a table of nodules counts...

# Detected vs Palpable
table(df.nodules$Detected,df.nodules$Palpable)

# Malignancy vs detection
table(df.nodules$BM,df.nodules$Detected)

# Size vs detection
table(df.nodules$Size,df.nodules$Detected)

# Malignancy vs Fatal
table(df.nodules$BM,df.nodules$Fatal)

# Detected vs Fatal (all nodules)
table(df.nodules$Detected,df.nodules$Fatal)

# Detected vs Fatal (malignant only)
table(df.nodules$Detected[df.nodules$BM==1],df.nodules$Fatal[df.nodules$BM==1])
nrow(df.nodules[df.nodules$BM==1,])


# ---------------------------
# ---- FIGURES AND TABLES----
# ---------------------------


# Figure 1 - Life expectancy histogram
hist(df.subjects$LifeExpect, breaks = 45)

# Figure 2 - Nodule size histogram
hist(df.nodules$Size, breaks=20)

# Figure 3 - Patients over time
plot(df.trends$Year, df.trends$Women, xlim=c(0,120), type='l', bty='n')
lines(df.trends$Year, df.trends$Men)
lines(df.trends$Year, df.trends$M_Women, lty=2)
lines(df.trends$Year, df.trends$M_Men, lty=2)
      
# Figure 4 - Look at the lag between creation date and detection date
x = df.nodules$YearFound[!is.na(df.nodules$YearFound)] - df.nodules$FlipAge[!is.na(df.nodules$YearFound)]
table(x)
hist(x, breaks=60)

# Figure 5 - Now let's focus on getting data, which will illustrate
# the increase in diagnoses (over and under) with increasing US resolution
# The basic structure starts with a count of nodules by size
table(df.nodules$Size)
table(df.nodules$Size[df.nodules$BM==0])
table(df.nodules$Size[df.nodules$Detected=='Y'])
table(df.nodules$Size[df.nodules$Detected=='Y' & df.nodules$BM==0])
table(df.nodules$Size[df.nodules$Detected=='N' & df.nodules$BM==0])
table(df.nodules$Size[df.nodules$Detected=='Y' & df.nodules$BM==1])
table(df.nodules$Size[df.nodules$Detected=='N' & df.nodules$BM==1])
table(df.nodules$Size,df.nodules$FNAstatus)
table(df.nodules$Size,df.nodules$FNAresult)
table(df.nodules$Size,df.nodules$USstatus)
table(df.nodules$Size,df.nodules$USresult)
table(df.nodules$Size[df.nodules$Detected=='Y' & (df.nodules$FNAstatus=='N' | df.nodules$USstatus=='N')])
table(df.nodules$Size,df.nodules$Predict)

# Table 2 - Malignant nodes and patient mortality
table(df.nodules$Type, df.nodules$Fatal)

# Table 4 - Prediction results
nrow(df.nodules[!is.na(df.nodules$FNAstatus), ])
table(df.nodules$FNAresult, df.nodules$BM)



