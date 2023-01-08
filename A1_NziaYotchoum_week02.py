# -*- coding: utf-8 -*-
 
##############################################
## Suggested Problems#########################
##############################################

 
 
##############################################
## Juypter Exercise Problems    ##############
##############################################

### lesson 3 Exercise 1 (3-1)
print("Compute the sum of the integers up to and including n")

sums = sum(range(51))

n=50
formula = (n*n-n)/2
print(sums,formula)


### lesson 3 Exercise 2 (3-2)
print("##############################################")

### lesson 3 Exercise 3 (3-3)
print("##############################################")
let = ['BLANK','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
       'P','Q','R','S','T','U','V','W','X','Y','Z']
pts =      [0, 1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10]
numTiles = [2, 9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1]

word = 'quickly'
word = word.upper()
word= list(word)
total_points = 0
for i,char in enumerate(word):
  if char == " ":
    total_points += pts[let.index("BLANK")]
  else:
    total_points += pts[let.index(char)]
print(f"The word {''.join(c for c in word).lower()!r} is worth a total of {total_points} points")
 
##############################################
## Board Problems    #########################
##############################################
 
