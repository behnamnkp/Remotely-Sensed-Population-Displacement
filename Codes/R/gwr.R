library(spgwr)
library(tidyverse)
library(ggplot2)
library(maptools)
library(sf)
library(car)

ntl.data1 = read.csv('G:/backupC27152020/Population_Displacement_Final/Resources/Results/observations_median_ntl_annual_incorrected_03292020.csv')

print('****Land use only model + GWR *******************************************************************')
#
ntl.bw <-gwr.sel(Pop2013 ~ area_hr , data=ntl.data1, 
                 coords=cbind(ntl.data1$X, ntl.data1$Y),adapt=FALSE)
ntl.gauss <- gwr(Pop2013 ~ area_hr , data=ntl.data1,
                 coords=cbind(ntl.data1$X, ntl.data1$Y), bandwidth = 316.521, hatmatrix=TRUE, se.fit=TRUE)
print(ntl.gauss)
# 
results1<-as.data.frame(ntl.gauss$SDF)
results1$ntl_clip_id <- ntl.data1$ntl_clip_id_y
write.csv(results1, "G:/backupC27152020/Population_Displacement_Final/Resources/Results/GWR_lndus_03292020.csv") 


ntl.data1 = read.csv('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Results/observations_median_ntl_annual_incorrected_03292020.csv')

#print('****Non-spatial: Land use and NTL + simple OLS *******************************************************************')
#ntl <- lm(Pop2013 ~ NTL2013_hr + NTL2013_nr, data = ntl.data1)
#summary(ntl)

#print('**** Non-spatial: Land use and NTL + GWR *******************************************************************')
#
#ntl.bw <-gwr.sel(Pop2013 ~ NTL2013_hr + NTL2013_nr, data=ntl.data1, 
               #  coords=cbind(ntl.data1$X, ntl.data1$Y),adapt=FALSE)
#ntl.gauss <- gwr(Pop2013 ~ NTL2013_hr + NTL2013_nr, data=ntl.data1,
 #                coords=cbind(ntl.data1$X, ntl.data1$Y), adapt=ntl.bw, hatmatrix=TRUE, se.fit=TRUE)
#print(ntl.gauss)

# 
#results2<-as.data.frame(ntl.gauss$SDF)
#results2$ntl_clip_id <- ntl.data1$ntl_clip_id_y
#write.csv(results2, "G:/backupC27152020/Population_Displacement_Final/Resources/Results/GWR_median_ntlhrnr_annual_incorrected_03292020.csv") 

print('**** spatial: Land use and NTL + GWR *******************************************************************')
#
ntl.bw <-gwr.sel(Pop2013 ~ NTL2013_hr, data=ntl.data1, 
                 coords=cbind(ntl.data1$X, ntl.data1$Y),adapt=FALSE)
ntl.gauss <- gwr(Pop2013 ~ NTL2013_hr, data=ntl.data1,
                 coords=cbind(ntl.data1$X, ntl.data1$Y), bandwidth = 240.6537 , hatmatrix=TRUE, se.fit=TRUE)
print(ntl.gauss)

# 
results2<-as.data.frame(ntl.gauss$SDF)
results2$ntl_clip_id <- ntl.data1$ntl_clip_id_y
write.csv(results2, "C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Results/GWR_median_ntlhr_annual_incorrected_03292020.csv") 

print('**** Non-spatial: Land use and NTL + GWR *******************************************************************')
lregression = lm(Pop2013 ~ NTL2013_hr , data=ntl.data1)
summary(lregression)
x <- as.data.frame(ntl.data1$ntl_clip_id_y)
x$pred <- lregression$fitted.values

write.csv(x, "C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Results/LR_median_ntlhr_annual_incorrected_03292020.csv") 

#car::vif(lregression)#
#print('**** Non-spatial: Land use and NTL + GWR *******************************************************************')
#
#ntl.bw <-gwr.sel(Pop2013 ~ area_hr + CNTL2013, data=ntl.data1, 
         #        coords=cbind(ntl.data1$X, ntl.data1$Y),adapt=FALSE)
#ntl.gauss <- gwr(Pop2013 ~ area_hr + CNTL2013, data=ntl.data1,
             #    coords=cbind(ntl.data1$X, ntl.data1$Y), adapt=ntl.bw, hatmatrix=TRUE, se.fit=TRUE)
#print(ntl.gauss)

# 
#results2<-as.data.frame(ntl.gauss$SDF)
#results2$ntl_clip_id <- ntl.data1$ntl_clip_id_y
#write.csv(results2, "G:/backupC27152020/Population_Displacement_Final/Resources/Results/GWR_median_ntl_annual_incorrected_03292020.csv") 



#intersect = read.csv('G:/backupC27152020/Population_Displacement_Final/Resources/temp/test.csv')
#intersect.bw <-gwr.sel(disPop2013_prime ~ disNTL_prime2013 + HR + LR + NR, data=intersect, 
    #             coords=cbind(intersect$X_intersect, intersect$Y_intersect),adapt=FALSE)
#ntl.gauss <- gwr(disPop2013_prime ~ disNTL_prime2013+ HR + LR + NR, data=intersect,
         #        coords=cbind(intersect$X_intersect, intersect$Y_intersect), bandwidth = 92.43835, hatmatrix=TRUE, se.fit=TRUE)
#print(ntl.gauss)

# Association
library(fastmatrix)
library(SpatialPack)
# Murray Smelter site dataset
data(murray)

# defining the arsenic (As) and lead (Pb) variables from the murray dataset
x <- murray$As
y <- murray$Pb
# extracting the coordinates from Murray dataset
coords <- murray[c("xpos","ypos")]
# computing the codispersion coefficient
z <- codisp(x, y, coords)
z

## plotting the codispersion coefficient vs. the lag distance
plot(z)
# Comovement between two time series representing the monthly deaths
# from bronchitis, emphysema and asthma in the UK for 1974-1979
x <- mdeaths
y <- fdeaths
coords <- cbind(1:72, rep(1,72))
z <- codisp(x, y, coords)
# plotting codispersion and cross-correlation functions
par(mfrow = c(1,2))
ccf(x, y, ylab = "cross-correlation", max.lag = 20)
plot(z)


# defining the arsenic (As) and lead (Pb) variables from the murray dataset
x <- murray$As
y <- murray$Pb
# extracting the coordinates from Murray dataset
coords <- murray[c("xpos","ypos")]
# computing Tjostheim's coefficient
z <- cor.spatial(x, y, coords)
z

data(texmos2)
y <- imnoise(texmos2, type = "gaussian")
plot(as.raster(y))
o <- CQ(texmos2, y, h = c(0,1))
o

y <- imnoise(texmos2, type = "speckle")
plot(as.raster(y))
o <- CQ(texmos2, y, h = c(0,1))
o

# Murray Smelter site dataset
data(murray)
# defining the arsenic (As) and lead (Pb) variables from the murray dataset
x <- murray$As
y <- murray$Pb
# extracting the coordinates from Murray dataset
coords <- murray[c("xpos","ypos")]
# computing the modified t-test of spatial association
z <- modified.ttest(x, y, coords)
z
# display the upper bounds, cardinality and the computed Moran's index
summary(z)


# 
returnees_cross_correlation = read.csv('G:/backupC27152020/Population_Displacement_Final/Resources/Field/returnees_cross_correlation.csv')

# 2017

x2017 <- returnees_cross_correlation$estpop2017change
y_returnee_displaced_2017 <- returnees_cross_correlation$average_returnee_displaced_2017
y_returnee_2017 <- returnees_cross_correlation$average_returnee_2017

coords <- returnees_cross_correlation[c("coord.x","coord.y")]

cor_returnee_displaced_2017 = cor(returnees_cross_correlation$estpop2017change,returnees_cross_correlation$average_returnee_displaced_2017)
cor_returnee_displaced_2017
z_returnee_displaced_2017 <- modified.ttest(x2017, y_returnee_displaced_2017, coords)
z_returnee_displaced_2017
summary(z_returnee_displaced_2017)

cor_returnee_2017 = cor(returnees_cross_correlation$estpop2017change,returnees_cross_correlation$average_returnee_2017)
cor_returnee_2017
z_returnee_2017 <- modified.ttest(x2017, y_returnee_2017, coords)
z_returnee_2017
summary(z_returnee_2017)

#2018
x2018 <- returnees_cross_correlation$estpop2018change
y_returnee_displaced_2018 <- returnees_cross_correlation$average_returnee_displaced_2018
y_returnee_2018 <- returnees_cross_correlation$average_returnee_2018

coords <- returnees_cross_correlation[c("coord.x","coord.y")]

cor_returnee_displaced_2018 = cor(returnees_cross_correlation$estpop2018change,returnees_cross_correlation$average_returnee_displaced_2018)
cor_returnee_displaced_2018
z_returnee_displaced_2018 <- modified.ttest(x2018, y_returnee_displaced_2018, coords)
z_returnee_displaced_2018
summary(z_returnee_displaced_2018)

cor_returnee_2018 = cor(returnees_cross_correlation$estpop2018change,returnees_cross_correlation$average_returnee_2018)
cor_returnee_2018
z_returnee_2018 <- modified.ttest(x2018, y_returnee_2018, coords)
summary(z_returnee_2018)


