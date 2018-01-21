# install.packages("wordcloud")
library(wordcloud)
bluePalette <- brewer.pal(n = 9, "Blues") # For these palettes, the max is 9

dic <- read.csv("dic.csv")
dic <- dic %>% 
  filter(complete.cases(dic))


#new data cloud
for(i in 1:length(dic[,1])) {
  dic$weight[i] <- round(runif(1, 10, 100),0)
}


str(dic)

dic <- dic %>% 
  rename(word = dendrogram) %>% 
  rename(fre= weight)


library(RColorBrewer)
bluePalette <- brewer.pal(n = 9, "Blues")
greyPalette <- brewer.pal(n = 9, "Greys")
rainbow <- colorRampPalette(c("black", "red"))( 500) 

wordcloud2(dic,  figPath = "C:/Users/DBers/Documents/BlogPlatform/blog/facebookPage/minus3.png", # brandenburg-gate.png #minus.png 
           size = 0.4 ,color = c(bluePalette,  greenPalette, redPalette, rainbow))


