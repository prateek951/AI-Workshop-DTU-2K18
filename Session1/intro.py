# Integers and strings
my_var = 1

print(type(my_var))

my_var = "This is a string"
print(my_var)
print(type(my_var))

print("This is" + 'a new string')
print('-'*80)

# Strings
s = 'This is a new string'
# Print the length of the string
print(s.__len__())
print(s.lower())
print(s.lower().split('s'))


# Lists
l = [1.0,'ax',[1,2,323,'111'],0.1]
print(l[-1])
print(l)
print(int(l[2][-1]))
print(ord('a'),chr(98))
print('-'*80)
nums = [[1,5],[4,2],[7,3],[0,-1]]
print(nums)
print(sorted(nums))

print(sorted(nums,key=lambda x: [x[1],x[0]]))
nums.append(12)
nums.append(100)
print(nums)


# Dictionaries
d = {
    2 : 'val',
    'k' : 0.1,
    'q' : [0,{0:10,'k':1200}]
}
print(d.setdefault('my_key','qwqwqww'))
d['qqq'] = 1221
print(d.values())


# Tuples
a = (1,2,3)
print(a)

# Prints unordered collection of unique elements
p = set([1,3,3,2,2,3,4,5,2,1,1,1,2])
q = set([2,2,23,332,213,123,123,23])
print(p)
print(q)

x= 20
if x>5 and x<12:
    print("YES")
elif x>=12 and x<20:
    print('MED')
else:
    print("NO")

for ix in range(len(l)):
    print(ix,l[ix])

for ix in l:
    print(ix)
print('-'*80)
i=0
while i<10:
    print(i)
    i += 1

# Fizzbuzz
r = range(1,10)
for ix in r: 
    if ix%3==0:
        if ix%5==0:
            print("FizzBuzz")
        else:
            print("Fizz")
    elif ix%5==0:
        print('Buzz')
    else:
        print(ix)

def avg(*args,**kwargs):
    print(args)
    print(kwargs)

print(avg(12.01,12,a=10,b=20,pq=-3))

from math import sqrt as sq 

print(sq(2))

# Working with classes in python

class MyClass:
    """docstring for MyClass."""
    x=100
    def __init__(self,x=0.1,y=10):
        
        self.x = x 
        self.y = y 

    def create_new_z(self,z=11):
        self.z = z         
    def self_square(self):
        self.x = self.x**2 
        self.y = self.y**2
        self.z = self.z**2
    def multiply_w_sq(self,p):
        self.self_square()
        return p*(self.x) + p*(self.y) + p*(self.z)

a= MyClass()
print(a)
print(a.x)
print(a.y)
a.create_new_z()
print(a.z)
print(a.multiply_w_sq(-10))