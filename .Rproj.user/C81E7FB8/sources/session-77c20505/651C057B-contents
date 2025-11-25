library(readr)
library(dplyr)
library(ggplot2)



# 1. Frequency table
district_counts <- traceData_final %>%
  count(District, name = "Count") %>%
  arrange(desc(Count))

# 2. Percentage table
district_counts <- district_counts %>%
  mutate(Percentage = (Count / sum(Count)) * 100)

district_counts

# 3. Bar chart (Counts)
ggplot(district_counts, aes(x = reorder(District, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "coral3") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  ) +
  labs(
    title = "District Distribution - Counts",
    x = "District",
    y = "Participants Count"
  )

# 4. Bar chart (Percentages)
ggplot(district_counts, aes(x = reorder(District, -Percentage), y = Percentage)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "District Distribution - Percentages",
       x = "District", y = "Percentage (%)")

# 5. Chi-square test (District vs recency Interpretation)
cont_table <- table(traceData_final$District, traceData_final$`recency_interpretation`)
chisq.test(cont_table)

# OrgUnits

traceData_final <- read_csv("TRACE_Data.csv")

# Count org units per district
org_counts <- traceData_final %>%
  group_by(District) %>%
  summarise(Unique_OrgUnits = n_distinct(`orgUnit`)) %>%
  arrange(desc(Unique_OrgUnits))

org_counts

# Total org units
total_orgunits <- traceData_final %>%
  summarise(total = n_distinct(`orgUnit`))

total_orgunits

# Bar chart: Mapping OrgUnits per District
ggplot(org_counts, aes(x = reorder(District, -Unique_OrgUnits), 
                       y = Unique_OrgUnits)) +
  geom_bar(stat = "identity", fill = "coral3") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Mapping of Unique Organisation Units per District",
    x = "District",
    y = "Number of Health Facilities"
  )


# Bar chart with count labels
ggplot(org_counts, aes(x = reorder(District, -Unique_OrgUnits),
                       y = Unique_OrgUnits)) +
  geom_bar(stat = "identity", fill = "deeppink4") +
  geom_text(aes(label = Unique_OrgUnits),
            vjust = -0.3, size = 4) +     # label position + size
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Mapping of Unique Organisation Units per District",
    x = "District",
    y = "Number of Health Facilities"
  )

# Frequency + percentage
occ_counts <- traceData_final %>%
  count(Occupation_simple, name = "Count") %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  arrange(desc(Count))
          
ggplot(occ_counts, aes(x = reorder(Occupation_simple, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "grey0") +
  geom_text(aes(label = Count), vjust = -0.3, size = 4) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Occupation Distribution",
    x = "Occupation",
    y = "Count"
  )

# fixing typing errors
traceData_final <- traceData_final %>%
  mutate(traceData_final = ifelse(Occupation_simple == "Not working", 
                             "Not Working", 
                             Occupation_simple))


### Marriage Age and Participant Age


# Convert to long format for ggplot
# Convert the two age variables into long format
age_long <- traceData_final %>%
  select(age, marriage_age) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variable",
    values_to = "Age"
  )

# Boxplot
ggplot(age_long, aes(x = Variable, y = Age)) +
  geom_boxplot(fill = "green") +
  labs(
    title = "Boxplot of Participant Age and Marriage Age",
    x = "Variable",
    y = "Age"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 12),
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

# Filter relevant columns and remove missing values
traceData_final_plot <- traceData_final %>%
  select(District, `marriage_age`) %>%
  drop_na()

# Create side-by-side district boxplot
ggplot(traceData_final_plot, aes(x = District, y = `marriage_age`)) +
  geom_boxplot(fill = "coral3") +
  labs(
    title = "Marriage Age by District",
    x = "District",
    y = "Marriage Age"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  )

traceData_final_plot <- traceData_final %>%
  select(District, `age`) %>%
  drop_na()

# Create side-by-side district boxplot
ggplot(traceData_final_plot, aes(x = District, y = `age`)) +
  geom_boxplot(fill = "darkseagreen") +
  labs(
    title = "Age by District",
    x = "District",
    y = "Age"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  )

# Pregnancy
occ_counts <- traceData_final %>%
  count(pregnant_simple, name = "Count") %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  arrange(desc(Count))

ggplot(occ_counts, aes(x = reorder(pregnant_simple, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "magenta") +
  geom_text(aes(label = Count), vjust = -0.3, size = 4) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Pregnancy Status Distribution",
    x = "Pregnancy Status",
    y = "Count"
  )


# Recency Interpretation
occ_counts <- traceData_final %>%
  count(recency_interpretation, name = "Count") %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  arrange(desc(Count))

ggplot(occ_counts, aes(x = reorder(recency_interpretation, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "red") +
  geom_text(aes(label = Count), vjust = -0.3, size = 4) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Recency Interpretation Distribution",
    x = "Recency Interpretation",
    y = "Count"
  )


### Heat Map

# Select only numeric variables
traceData_final_numeric <- traceData_final %>% 
  select(where(is.numeric))

# Compute correlation matrix
cor_mat <- cor(traceData_final_numeric, use = "pairwise.complete.obs")

# Melt matrix for ggplot
cor_melt <- melt(cor_mat)

# Heatmap
ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low = "blue",
    high = "red",
    mid = "white",
    midpoint = 0,
    limit = c(-1, 1),
    space = "Lab",
    name = "Correlation"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10)
  ) +
  labs(
    title = "Heatmap of Variable Correlations",
    x = "",
    y = ""
  )

