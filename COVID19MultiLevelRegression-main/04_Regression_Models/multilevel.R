
# ##### data #####

# Read data
covid <-read.csv(file="RData.csv", header=TRUE,sep=",")
cols <- c("I1", "I2", "I3", "I4", "I5", "I6", "I7")
covid[cols] <- lapply(covid[cols], factor)  #set indicator var as factor
countries = c("Australia", "Canada", "United Kingdom", "United States")

# ##### data (split period) #####

# Read data
covid2 <-read.csv(file="RData_s.csv", header=TRUE,sep=",")
covid2[cols] <- lapply(covid2[cols], factor)  #set indicator var as factor

# ##### 1a/ basic model: for each country #####
for (c in countries){
      print(paste("******************************", c, "***********************************"))
      #data of that country
      assign(c,subset(covid, country==c))
  
      #list of I that have >1 levels
      formula = "~ "
      for (p in cols){
          if (length(unique(get(c)[[p]]))>1){
            add = as.character(p)
            formula = paste(formula, add, "+")
          }
      }
      formula = substr(formula,1,nchar(formula)-2)

      # ----- 1a.1/ basic model (sentiment only) -----
      print('#################### sentiment only ####################')
      #sentiment only (first dev.)
      Model_S1 <- lm(firstDev ~ s1+s2+s3+s4+s5+s6+s7, data = get(c))
      print(summary(Model_S1))
      print(AIC(Model_S1))
      assign(paste("Model_S1_", c, sep=""), Model_S1)
      #sentiment only (second dev.)
      Model_S2 <- lm(secondDev ~ s1+s2+s3+s4+s5+s6+s7, data = get(c))
      print(summary(Model_S2))
      print(AIC(Model_S2))
      assign(paste("Model_S2_", c, sep=""), Model_S2)
      

      # ----- 1a.2/ basic model (Implementation only) -----
      print('#################### Implementation only ####################')
      #I only (first dev.)
      Model_I1 <- lm(as.formula(paste("firstDev", formula)), data = get(c))
      print(summary(Model_I1))
      print(AIC(Model_I1))
      assign(paste("Model_I1_", c, sep=""), Model_I1)
      #I only (second dev.)
      Model_I2 <- lm(as.formula(paste("secondDev", formula)), data = get(c))
      print(summary(Model_I2))
      print(AIC(Model_I2))
      assign(paste("Model_I2_", c, sep=""), Model_I2)
      
      # ----- 1a.3/ basic model (sentiment & Implementation) -----
      print('#################### Sentiment & Implementation ####################')
      #sentiment only (first dev.)
      Model_SI1 <- lm(paste("firstDev", formula, "+s1+s2+s3+s4+s5+s6+s7"), data = get(c))
      print(summary(Model_SI1))
      print(AIC(Model_SI1))
      assign(paste("Model_SI1_", c, sep=""), Model_SI1)
      #sentiment only (second dev.)
      Model_SI2 <- lm(paste("secondDev", formula, "+s1+s2+s3+s4+s5+s6+s7"), data = get(c))
      print(summary(Model_SI2))
      print(AIC(Model_SI2))
      assign(paste("Model_SI2_", c, sep=""), Model_SI2)
}




# ##### 1b/ basic model: all countries tgt #####
    # ----- 1b.1/ basic model (sentiment only) -----
    #sentiment only (first dev.)
    Model_S1 <- lm(firstDev ~ s1+s2+s3+s4+s5+s6+s7, data = covid)
    summary(Model_S1)
    AIC(Model_S1)
    #sentiment only (second dev.)
    Model_S2 <- lm(secondDev ~ s1+s2+s3+s4+s5+s6+s7, data = covid)
    summary(Model_S2)
    AIC(Model_S2)
    
    # ----- 1b.2/ basic model (Implementation only) -----
    #sentiment only (first dev.)
    Model_I1 <- lm(firstDev ~ I1+I2+I3+I4+I5+I6+I7, data = covid)
    summary(Model_I1)
    AIC(Model_I1)
    #sentiment only (second dev.)
    Model_I2 <- lm(secondDev ~ I1+I2+I3+I4+I5+I6+I7, data = covid)
    summary(Model_I2)
    AIC(Model_I2)
    
    # ----- 1b.3/ basic model (sentiment & Implementation) -----
    #sentiment only (first dev.)
    Model_SI1 <- lm(firstDev ~ s1+s2+s3+s4+s5+s6+s7+I1+I2+I3+I4+I5+I6+I7, data = covid)
    summary(Model_SI1)
    AIC(Model_SI1)
    #sentiment only (second dev.)
    Model_SI2 <- lm(secondDev ~ s1+s2+s3+s4+s5+s6+s7+I1+I2+I3+I4+I5+I6+I7, data = covid)
    summary(Model_SI2)
    AIC(Model_SI2)
    

    

# ##### 2/ multilevel models #####
# Load lme4, MuMIn package
library(lme4)
library(MuMIn)

    # ----- 2ai/ varying intercept -----
    #first derivative as y
    tpb_i1 <- lmer(firstDev ~ I1+I2+I3+I4+I5+I7 + (1|country), 
                  data = covid)
    summary(tpb_i1)
    coef(tpb_i1)$country  #coef for random effects
    
    #second derivative as y
    tpb_i2 <- lmer(secondDev ~ I1+I2+I3+I4+I5+I7 + (1|country), 
                   data = covid)
    summary(tpb_i2)
    coef(tpb_i2)$country #coef for random effects
    
    
    
    #variance significance (type II anova - for model w/o interaction term)
    car::Anova(tpb_i1)
    car::Anova(tpb_i2)
    
    #model fit
      #r-squared
      r.squaredGLMM(tpb_i1) 
      r.squaredGLMM(tpb_i2)
          #"R2m" explained by fixed effects; "R2c" explained by fixed & random effects
      #AIC
      AIC(tpb_i1)
      AIC(tpb_i2)


    
    # ----- 2aii/ varying intercept (split periods) -----
    #first derivative as y
    tpb_i1s <- lmer(firstDev ~ I1+I2+I3+I4+I5+I7 + (1|country), 
                    data = covid2)
    summary(tpb_i1s)
    coef(tpb_i1s)$country  #coef for random effects
    
    #second derivative as y
    tpb_i2s <- lmer(secondDev ~ I1+I2+I3+I4+I5+I7 + (1|country), 
                    data = covid2)
    summary(tpb_i2s)
    coef(tpb_i2s)$country #coef for random effects
    
    
    #variance significance (type II anova - for model w/o interaction term)
    car::Anova(tpb_i1s)
    car::Anova(tpb_i2s)
    
    #model fit
      #r-squared
      r.squaredGLMM(tpb_i1s) 
      r.squaredGLMM(tpb_i2s)
      #"R2m" explained by fixed effects; "R2c" explained by fixed & random effects
      #AIC
      AIC(tpb_i1s)
      AIC(tpb_i2s)
      
    
    # ----- 2bi/ varying slope -----
    #first derivative as y
    tpb2.1 <- lmer(firstDev ~ 1+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                   data = covid)
    summary(tpb2.1)
    coef(tpb2.1)$country  #coef for random effects

    #second derivative as y
    tpb2.2 <- lmer(secondDev ~ 1+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                   data = covid)
    summary(tpb2.2)
    coef(tpb2.2)$country #coef for random effects
    
    
    #model fit
      #r-squared
      r.squaredGLMM(tpb2.1) 
      r.squaredGLMM(tpb2.2)
      #"R2m" explained by fixed effects; "R2c" explained by fixed & random effects
      #AIC
      AIC(tpb2.1)
      AIC(tpb2.2)
    
    # ----- 2bii/ varying slope (split periods) -----
    #first derivative as y
    tpb2.1s <- lmer(firstDev ~ 1+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                    data = covid2)
    summary(tpb2.1s)
    coef(tpb2.1s)$country  #coef for random effects
    
    #second derivative as y
    tpb2.2s <- lmer(secondDev ~ 1+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                    data = covid2)
    summary(tpb2.2s)
    coef(tpb2.2s)$country #coef for random effects
    
    
    #model fit
    #r-squared
    r.squaredGLMM(tpb2.1s) 
    r.squaredGLMM(tpb2.2s)
    #"R2m" explained by fixed effects; "R2c" explained by fixed & random effects
    #AIC
    AIC(tpb2.1s)
    AIC(tpb2.2s)
    
    # ----- 2ci/ varying intercepts & slope-----
      #Correlated random intercepts and slopes
          #first derivative as y 
          tpb3.1 <- lmer(firstDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                         data = covid)
          #second derivative as y
          tpb3.2 <- lmer(secondDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                           data = covid)
      #independent random intercepts and slopes
          #first derivative as y 
          tpb3.1i <- lmer(firstDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7||country), 
                           data = covid)
          #second derivative as y
          tpb3.2i <- lmer(secondDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7||country), 
                           data = covid)
      #model fit
      r.squaredGLMM(tpb3.1) 
      r.squaredGLMM(tpb3.2) 
      r.squaredGLMM(tpb3.1i) 
      r.squaredGLMM(tpb3.2i) 
      
      AIC(tpb3.1)
      AIC(tpb3.2)
      AIC(tpb3.1i)
      AIC(tpb3.2i)
      
      
    # ----- 2cii/ varying intercepts & slope (split periods)-----
    #Correlated random intercepts and slopes
    #first derivative as y 
    tpb3.1s <- lmer(firstDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                    data = covid2)
    #second derivative as y
    tpb3.2s <- lmer(secondDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7|country), 
                    data = covid2)
    #independent random intercepts and slopes
    #first derivative as y 
    tpb3.1i_s <- lmer(firstDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7||country), 
                      data = covid2)
    #second derivative as y
    tpb3.2i_s <- lmer(secondDev ~ I1+I2+I3+I4+I5+I6+I7+ (0+I1+I2+I3+I4+I5+I6+I7||country), 
                      data = covid2)
    #model fit
    r.squaredGLMM(tpb3.1s) 
    r.squaredGLMM(tpb3.2s) 
    r.squaredGLMM(tpb3.1i_s) 
    r.squaredGLMM(tpb3.2i_s) 
    
    AIC(tpb3.1s)
    AIC(tpb3.2s)
    AIC(tpb3.1i_s)
    AIC(tpb3.2i_s)

    
    # ----- 2/ overall best multilevel models -----
    summary(tpb3.1i)
    coef(tpb3.1i)$country
    
    #level 2 model fitting
  
    # Read data
    lv2 <-read.csv(file="lv2.csv", header=TRUE,sep=",")
    
    for (c in countries){
      print(paste("******************************", c, "***********************************"))
      #data of that country
      assign(c,subset(lv2, country==c))

      #sentiment only (first dev.)
      Model <- lm(y ~ + offset(a) + SN+A+PBC+O, data = get(c))
      print(summary(Model))
      print(AIC(Model))
      assign(paste("Model_", c, sep=""), Model)
     
    }
    
    
    
    
