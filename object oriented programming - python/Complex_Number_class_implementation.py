class Complex(object):
    ''' Class of complex number which instances are of form a +bj, where a is  a real part
    of the number and b is an imaginary part of the number. j means imaginary part and stands
    for sqrt(-1). All instances of Complex class support basic math operations like
    + , - , * , / , += , -= , /= , == , != '''

    def __init__(self, real_ = 0.0, img_ = 0.0):
        try:
            self.__real = float(real_)
            self.__img = float(img_)
        except ValueError:
            print  "Expected real number, not character or symbol\n"
            raise
        except TypeError:
            print 'When setting the real nmber part provide just a rael number\n'
            raise


    def __repr__(self):
        if self.__img < 0:
            return str(self.__real) + "" + str(self.__img) + "j"
        else:
            return str(self.__real) + "+" + str(self.__img) + "j"

    def __str__(self):
        if self.__img < 0:
            return "(" + str(self.__real) + "" + str(self.__img) + "j)"
        if not self.__real:
            return "(" + str(self.__img) + "j)"
        else:
            return str(self.__real) + "+" + str(self.__img) + "j"


    def __add__(self, other_):
        ''' calculates sum of two complex numbers'''
        return  Complex(self.__real + other_.__real , self.__img + other_.__img)

    def __iadd__(self, other_):
        ''' Add one complex number to the other '''
        self.__real += other_.__real
        self.__img += other_.__img
        return  self  # return self but not new instance like in __add__

    def __sub__(self, other_):
        ''' substract one complex number from the other '''
        return Complex(self.__real - other_.__real , self.__img - other_.__img)

    def __isub__(self, other_):
        ''' substract one complex number from the other '''
        self.__real -= other_.__real
        self.__img -= other_.__img
        return  self  # return self but not new instance like in __sub__


    def __mul__(self, other_):
        ''' multiplicate two complex numbers '''
        return (self.__real * other_.__real +  self.__img * other_.__img * (-1),
                self.__real * other_.__img + self.__img * other_.__real)

    def __imul__(self, other_):
        ''' multiply one complex number by the other '''
        self.__real , self.__img = (self.__real * other_.__real + self.__img * other_.__img)\
        , (self.__real * other_.__img + self.__img * other_.__real)
        return self

    def __div__(self, other_):
        ''' devide one complex number by other according to special laws'''
        real = self.__real * other_.__real + self.__img * other_.__img
        imag = self.__real * other_.__img *(-1) + self.__img * other_.__real
        denominator = other_.__real * other_.__real + other_.__img * other_.__img
        if denominator:
            return Complex(real/denominator , imag /denominator)
        else:
            raise ZeroDivisionError

    def __idiv__(self, other_):
        ''' devide one complex number by other complex number'''
        self.__real = (self.__real * other_.__real + self.__img * other_.__img)  / (other_.__real * other_.__real + other_.__img * other_.__img)
        self.__img = (self.__real * other_.__img *(-1) + self.__img * other_.__real) /(other_.__real * other_.__real + other_.__img * other_.__img)

        if other_.__img or other_.__real:
            return self
        else:
            raise ZeroDivisionError

    def __eq__(self, other_):
        ''' return True if real parts and img parts both are equal'''
        return self.__real == other_.__real and self.__img == other_.img


    def __ne__(self, other_):
        ''' return True if at least one of the parts is not equal'''
        return  self.__real != other_.__real and self.__img != other_.img

    @property
    def real(self):
        ''' Retrives the value of real part of the complex number'''
        return self.__real

    @property
    def img(self):
        ''' Retrives the value of real part of the complex number'''
        return self.__img

    @real.setter
    def real(self,val_):
        ''' sets value to the real part of the complex number '''
        try:
            self.__real = float(val_)
        except ValueError:
            print " Expected real number, ot character or symbol"
            raise
        except TypeError:
            print " When setting the real part provide just a real number"
            raise

    @img.setter
    def img(self, val_):
        '''  sets value to the img part of the complex number '''
        try:
            self.__img = float(val_)
        except ValueError:
            print " Expected real number, ot character or symbol"
            raise
        except TypeError:
            print " When setting the img part provide just a real number"
            raise


#####################   Test   ####################################

a = Complex()
print a
b = Complex(3)
print b
c = Complex("6")
print c
d = Complex(3,2)
print d
e = Complex(4,-3)
print e

#error = Complex("a")
#print error

add = d + e
print add

sub = e - d
print sub

e += d
print e

d -= e
print d

mul = e * d
print mul

e *= d
print e

div = d / e
print div

div /= e
print div

print e.__eq__(d)
print e.__ne__(d)








































































