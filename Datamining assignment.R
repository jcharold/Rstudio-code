
#installing Libraries
library(worldfootballR)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)
library(ggthemes)
library(skimr)
library(dplyr) 
library(tidyr)
library(caret)

###------------------------------------------------------------------------------------------------------------------------

###Fetching data for analysis

big5_player_shooting <- fb_big5_advanced_season_stats(season_end_year= c(2024), stat_type= "shooting", team_or_player= "player")
big5_player_shooting <- big5_player_shooting %>% filter(Pos != "MF,DF" & Pos != "DF,MF" & Pos != "DF,FW")
big5_player_shooting <- big5_player_shooting %>% filter(Pos != "GK" & Pos != "DF" & Pos != "MF")
View(big5_player_shooting)

#get big5_player_standard
big5_player_standard <- fb_big5_advanced_season_stats(season_end_year= c(2024), stat_type= "standard", team_or_player= "player")
big5_player_standard <- big5_player_standard %>% filter(Pos != "MF,DF" & Pos != "DF,MF" & Pos != "DF,FW")
big5_player_standard <- big5_player_standard %>% filter(Pos != "GK" & Pos != "DF" & Pos != "MF")
View(big5_player_standard)

#get big5_player_possession
big5_player_possession <- fb_big5_advanced_season_stats(season_end_year= c(2024), stat_type= "possession", team_or_player= "player")
View(big5_player_possession)

merged_data <- big5_player_standard %>%
  left_join(big5_player_possession, by = c("Player", "Age", "Squad", "Season_End_Year", "Pos","Nation","Comp","Born"))

merged_data <- merged_data %>%
  left_join(big5_player_shooting, by = c("Player", "Age", "Squad", "Season_End_Year", "Pos","Nation","Comp","Born"))


###----------------------------------------------------------------------------
### STEP 2: Player Success Analysis 

table <- fb_season_team_stats(country = "ENG", 
                              gender = "M", 
                              season_end_year = 1992:2023, 
                              tier = "1st", 
                              stat_type = "league_table")

model <- lm(Pts ~ GF, data = table)
summary(model)
# Coefficient is 0.96330
# R-squared is 0.7794

###----------------------------------------------------------------------------

###Step 3: Identifying Critical Attributes 
#RFE ANALYSIS 

#Creating the dummy for G+A 
merged_data$exPts <- merged_data$`G+A` * 0.96330 #CONVERTS G+A INTO POINTS 
merged_data$exPts_dummy <- ifelse(merged_data$exPts >= 12, 1, 0) 

#Manually pick which predictors make sense to include for testing relevance
predictor_columns <- unique(c(
  "Att 3rd_Touches", "Succ_Take", "PrgDist_Carries", "CPA_Carries", 
  "PrgR_Receiving", "SoT_per_90_Standard", "Dist_Standard", 
  "Sh_per_90_Standard", "SoT_percent_Standard", 
  "Att Pen_Touches", "Succ_percent_Take", "Final_Third_Carries", 
  "Live_Touches", "Min_Playing"
))

#Create new dataframe with only the selected predictors and add the exPts_dummy column
merged_data_short <- merged_data %>% select(all_of(predictor_columns), exPts_dummy) %>% na.omit()

# Set up Recursive Feature Elimination (RFE) with cross-validation
set.seed(123)
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)  # 10-fold cross-validation

# Convert exPts_dummy to factor (if not already done) to ensure classification is recognized
merged_data_short$exPts_dummy <- as.factor(merged_data_short$exPts_dummy)

# Run RFE to select top predictors for classification
set.seed(12)
rfe_results <- rfe(
  merged_data_short[, predictor_columns], 
  merged_data_short$exPts_dummy, 
  sizes = c(5, 6, 7, 8, 10),  # Specify range of subset sizes
  rfeControl = control
)

# View the results and selected predictors
print(rfe_results)

#Relevant predictors:
#Att Pen_Touches, Min_Playing, SoT_per_90_Standard, Att 3rd_Touches, Live_Touches

#VALIDATION OF RFE USING LOGISTIC REGRESSION----------------------------------------------------------------------------
logistic_model <- glm(
  exPts_dummy ~ `Att 3rd_Touches` + Live_Touches + Min_Playing + SoT_per_90_Standard + Min_Playing + `Att Pen_Touches`,
  data = train.df,
  family = "binomial"
)


#Check accuracy of logistics model of the relevant predictors
#Split the merged dataset into a training and validation set
set.seed(123)
train.index <- sample(row.names(merged_data), 0.8 * nrow(merged_data))
valid.index <- setdiff(row.names(merged_data), train.index)

# Create training and validation datasets
train.df <- merged_data[train.index, ]
valid.df <- merged_data[valid.index, ]

predicted_probs <- predict(logistic_model, newdata = valid.df, type = "response")

predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Actual classes
actual_classes <- valid.df$exPts_dummy

# Confusion matrix
conf_matrix <- confusionMatrix(
  factor(predicted_classes, levels = c(0, 1)), # Predicted values as factor
  factor(actual_classes, levels = c(0, 1))     # Actual values as factor
)

###-----------------------------------------------------------------------------
### STEP 4: Create the optimal player based on relevant predictors


#remove observations if G+A is smaller than 13
merged_data <- merged_data %>% filter(`G+A` >= 13)

# Proceed with the pre-processing and k-NN steps
preprocess_model <- preProcess(merged_data %>%
                                 select(`Att Pen_Touches`, Live_Touches,`Att 3rd_Touches`, 
                                        SoT_per_90_Standard, Min_Playing), 
                               method = c("center", "scale"))

# Scale the Full Dataset
big5_scaled <- predict(preprocess_model, newdata = merged_data)

# Create Optimal Player Profile Using 80th Percentile Values
optimal_player <- data.frame(
  `Att 3rd_Touches` = quantile(merged_data$`Att 3rd_Touches`, 0.80, na.rm = TRUE),
  Live_Touches = quantile(merged_data$Live_Touches, 0.80, na.rm = TRUE),
  SoT_per_90_Standard = quantile(merged_data$SoT_per_90_Standard, 0.80, na.rm = TRUE),
  Min_Playing = quantile(merged_data$Min_Playing, 0.80, na.rm = TRUE),
  `Att Pen_Touches` = quantile(merged_data$`Att Pen_Touches`, 0.80, na.rm = TRUE)
)

# Rename columns in optimal_player to match preprocess_model exactly
colnames(optimal_player) <- colnames(preprocess_model$mean)

# Apply the preprocessing model to scale the optimal player profile
optimal_player_scaled <- predict(preprocess_model, newdata = optimal_player)

###-----------------------------------------------------------------------------
###Find closest player to nearest neighbor

library(FNN)

# SCALE data for NN
big5_player_knn_scaled <- big5_scaled %>% select(`Att Pen_Touches`, Live_Touches,`Att 3rd_Touches`, 
                                                 SoT_per_90_Standard, Min_Playing)

optimal_player_knn_scaled <- optimal_player_scaled %>% select(`Att Pen_Touches`, Live_Touches,`Att 3rd_Touches`, 
                                                              SoT_per_90_Standard, Min_Playing)

k <- 30
big5_player_knn_scaled <- na.omit(big5_player_knn_scaled)
optimal_player_knn_scaled <- na.omit(optimal_player_knn_scaled)

# Run the k-NN function
nearest_neighbors <- get.knnx(data = big5_player_knn_scaled, query = optimal_player_knn_scaled, k = k)

closest_players_indices <- nearest_neighbors$nn.index

# Retrieve closest players from merged_data, which has the G+A filter applied
closest_players <- merged_data[closest_players_indices, ]

# Only view Player, exPts_dummy and G+A and squad

closest_players <- closest_players %>% select(Player, exPts_dummy, `G+A`, Squad, Comp, Age)
closest_players

#remove players over 26
closest_players <- closest_players %>% filter(Age <= 26)
closest_players










