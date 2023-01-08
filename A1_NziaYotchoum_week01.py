# -*- coding: utf-8 -*-
import scipy.stats as stats
from math import sqrt
import math
##############################################
## Suggested Problems#########################
##############################################
# Exercise 1.1: 
print("##############################################")
print("Compute 1+1")
print (1+1, "\n")
# Exercise 1.2:
print("##############################################")
print("Hello, World!\n")

# Exercise 1.3:
print("##############################################")

# Exercise 1.4:
print("##############################################")
print("Convert from meters to British length units ")

length = 640 #length in meters

inches = length / 0.0254
feet = inches / 12
yard = feet / 3
mile = yard / 1760

print ("{} meters = {:.2f} inches = {:.2f} feet = {:.2f} yards = {:.4f} miles \n".format(length,inches,feet,yard,mile))

#Exercise 1.5:
print("##############################################")
print("Compute the mass of various substaces")


#Exercise 1.6
print("##############################################")
print("Compute the growth of money in a bank")

A = 1000 # initial Amount
n = 3 # number of years
p = .05 # interest rate

ammount = A * ( 1 + (p/100)) ** n

print("After {} years and a {} % interest rate, an initial amount of ${} has grown to ${:.2f} \n".format(n,p*100,A,ammount))
 

# Exercise 1.7: Find error(s) in a proram
print("##############################################")
print("Find errors")
print ("**Original** \n x=1; print 'sin(%g)=%g' % (x,sin(x)))")
#parenthesis were missing
x=1; print ('sin(%g)=%g' % (x,math.sin(x)),"\n")

#Exercise 1.10
print("##############################################")
def f(x):
    s = 2
    m = 0
    a = 1/((math.sqrt(2 * math.pi)) * s)
    t = -0.5*math.pow(((x-m)/s),2)
    e = math.exp(t)

    return a * e
print ("Evaluation of Guassian function with m = 0, s = 2, x = 1")
print (f'{f(1):.3f}', "\n")

#Exercise 1.11
print("##############################################")
print("Compute the Air Resistance on a football")
a= .11  #measurment in meters 11cm
g= 9.81 
m = 0.43
C = 0.4
A = math.pi * math.pow(a,2)
F = m*g
V = [33.33, 2.78]  #measruement in m/s
d = 1.2

print(f'The gravitional force on the football is {F:.2f} N')
print(f'The air resistance on the footbal with a velocity of {V[0]} m/s is {.5*C*d*A*V[0]**2:.3f} N')
print(f'The air resistance on the footbal with a velocity of {V[1]} m/s is {.5*C*d*A*V[1]**2:.3f} N \n')

#Exercise 1.11
print("##############################################")
print("How to cook the perfect egg")
M = [47 ,67]
p = 1.038
c = 3.7
K = 5.4*10**-3
ty = 70
tw = 100
Rt0 = 20
ft0 = 4
numerator = M[1]**(2/3)*c*p**(1/3)
denominator = K * math.pi**2*(4*math.pi/3)**(2/3)
flogarithm = math.log(0.76*((ft0-tw)/(ty-tw)))
rlogarithm = math.log(0.76*((Rt0-tw)/(ty-tw)))

ft = (numerator/denominator)* flogarithm
rt = (numerator/denominator)* rlogarithm

print(" Time it takes to cook an egg at room temp is {:.3f}".format(rt))
print(" Time it takes to cook an egg at fridge temp is {:.3f} \n".format(ft))
'''
Exercise 1.15: Explain why a program does not work
Find out why the following program does not work:
C = A + B
A = 3
B = 2
print C

The code above would not work because it is using two variables before they have been assigned
'''
print("##############################################")

'''
Exercise 1.16: Find errors in Python statements
Try the following statements in an interactive Python shell. Explain why
some statements fail and correct the errors.
'''
from math import tan
a1 = 2 
#a1 = b name 'b' is not defined
x = 2
y = x + 4 # is it 6?

print (tan(math.pi))
pi = 3.14159
print (tan(pi))
c = 4**3**2**3
_ = ((c-78564)/c + 32)
discount = .12 
AMOUNT = 120
amount = "120$"
address = "hpl@simula.no"
#ands = duck name 'duck' is not defined
classes = "INF1100, gr 2"
continue_ = x > 0
rev = fox = True
Norwegian = ['a human language']
true = "fox is rev in Norwegian"

print("##############################################")

a = 2; b = 1; c = 2
from math import sqrt
q = b*b - 4*a*c
q_sr = sqrt(abs(q))
x1 = (-b + q_sr)/2*a
x2 = (-b - q_sr)/2*a
print (x1, x2)


 
 
##############################################
## Juypter Exercise Problems    ##############
##############################################
 
### lesson 1 Exercise 1 (1-1)
print("##############################################")
print("Convert from meters to British length units ")

length = 640 #length in meters

inches = length / 0.0254
feet = inches / 12
yard = feet / 3
mile = yard / 1760

print ("{} meters = {:.2f} inches = {:.2f} feet = {:.2f} yards = {:.4f} miles \n".format(length,inches,feet,yard,mile))



### lesson 1 Exercise 2 (1-3)
print("##############################################")
print("Compute the growth of money in a bank")

A = 1000 # initial Amount
n = 3 # number of years
p = .05 # interest rate

ammount = A * ( 1 + (p/100)) ** n

print("After {} years and a {} % interest rate, an initial amount of ${} has grown to ${:.2f} \n".format(n,p*100,A,ammount))
 
### lesson 1 Exercise 3 (1-3)
print("##############################################")
print("Compute a confidence interval")


x_bar = 10
ci = 0.95 # should be 90% confidence interval
sx = 20
n_ = 100
val = sx / sqrt(n_)
tstar = stats.t.ppf(q=.05 , df=99 )
tval = (x_bar + tstar * val, x_bar - tstar * val )

print (f"({tval[0]:.3f},{tval[1]:.3f})"+"\n")
 
### lesson 2 Exercise 1(2-1)
print("##############################################")
print("Time of the day - using modulo")

totalsecs = 3700
hours = totalsecs // 3600
mins = (totalsecs % 3600) // 60
secs = totalsecs % 60

print( "{} total seconds equals {} hour(s), {} minute(s), and {} second(s)".format(totalsecs,hours,mins,secs)+"\n")

 
### lesson 2 Exercise 2 (2-2)
print("##############################################")
print("Theorem (Hull and Bobell)")

m = 15 
c = 4

a = 2 # initial 
found = False
while not found:
    q = (a-1)/3
    r = (a-1)/5
    if ".0" in str(q) and ".0" in str(r):
        found = True
    else:
        a+=1
print("m = {}, a = {}, c = {}".format(m,a,c))
i = 0;x = 0
print(f'{i:3}:{x:3}')
x = (a*x+c)%m
while x != 0:
    i+=1
    print(f'{i:3}:{x:3}')
    x = (a*x+c)%m
i+=1
print(f'{i:3}:{x:3}')
print("Period =",i)

### lesson 2 Exercise 3 (2-3)

print("##############################################")
print("Theorem (Hull and Bobell)")

m = 35 
c = 4

a = 2 # initial 
found = False
while not found:
    q = (a-1)/7
    r = (a-1)/5
    if ".0" in str(q) and ".0" in str(r):
        found = True
    else:
        a+=1
print("m = {}, a = {}, c = {}".format(m,a,c))
i = 0;x = 0
print(f'{i:3}:{x:3}')
x = (a*x+c)%m
while x != 0:
    i+=1
    print(f'{i:3}:{x:3}')
    x = (a*x+c)%m
i+=1
print(f'{i:3}:{x:3}')
print("Period =",i)


#################################
## Board Problems    #########################
##############################################
 
### 1) Lsn1 / Problem 4 - Babylonian Algorithm for finding the square root of a number
## Insert your working code
print("##############################################")
print("Babylonian Algorithm for finding the square root of a number")
import random as rn
num = 100
guess = rn.randint(1, 15)

diff = 1

while diff > 0.05:
    temp = (num/guess + guess)/2
    d1 = abs(temp - math.sqrt(num))
    d2 = abs(guess - math.sqrt(num))
    if d1 < d2:
        guess = temp
        diff = d1
    else:
        diff = d2
print("Actual sqrt of {} = {:.3f}, Estimate = {:.3f}, difference = {:.3f} \n ".format(num,math.sqrt(num),guess,diff))

print("##############################################")
print("Babylonian Algorithm for finding the cube root of a number")

### 2) Lsn1 / Problem 5 - Babylonian Algorithm for finding the cube root of a number
# Insert your working code
num = 8
guess = rn.randint(1, 15)

diff = 1

while diff > 0.05:
    temp = (num/math.pow(guess,2) + 2*guess)/3
    d1 = abs(temp - math.pow(num,1/3))
    d2 = abs(guess -  math.pow(num,1/3))
    if d1 < d2:
        guess = temp
        diff = d1
    else:
        diff = d2

print("Actual cube root of {} = {:.3f}, Estimate = {:.3f}, difference = {:.3f} \n ".format(num,math.pow(num,1/3),guess,diff))


###Lsn 2 / Problem 2
'''
Convert a number that is less than 256 from base 10 to base 2. Before proceeding to code, determine how many
digits you need
'''
print("##############################################")

val = 2
temp = val
binstr = ""
while temp >=1:
    binstr+= str(temp%2)
    temp//=2

print(f"{val} in binary is {binstr[::-1]}")

###Lsn 2 / Problem 3
print("##############################################")
a = 100; m = 21; c= 3
i = 0;x = 0
print(f'{i:3}:{x:3}')
x = (a*x+c)%m
while x != 0:
    i+=1
    print(f'{i:3}:{x:3}')
    x = (a*x+c)%m
i+=1
print(f'{i:3}:{x:3}')

print("Period =",i)



N = 30 
char = '*'
space =' '
for I in range(30):
    print(space*(N-1)+char+space*(N-1))
    char+="**"
    N-=1


n = 10
fact = 1
if n == 0 or n == 1:
    print(f"the factor of {n} is {fact} ")
else:
    temp = n
    while temp >=1:
        fact*=temp
        temp-=1
    print(f"the factor of {n} is {fact} ")