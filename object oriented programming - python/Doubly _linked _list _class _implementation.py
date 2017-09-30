class Dlist(object):

    def __init__(self, *args_):
        ''' a method which constructs a Dlist object each time an     
         instance of th class is created '''
        self.__head = self.__Node()
        self.__tail = self.__Node()
        self.__head.next = self.__tail
        self.__tail.prev = self.__head
        previous_node = self.__head
        self.__len = 0
        for arg in args_:
        # this part of the code go through all the elements
        # in the given list and creates a new node for each one of them and connects each node with his next and previous node
            current_node = self.__Node(previous_node,arg, previous_node.next)
            previous_node.next = current_node
            previous_node = current_node
            self.__len += 1
        if self.__len > 0:
            self.__tail.prev = current_node



    def __getitem__(self,index_):
        ''' method which gets an index and returns the value stored in the given index in the list '''
        counter = 0
        current_node = self.__head.next
        if self.__len ==0:
            return " the list is empty "
        elif index_ > self.__len:
            raise IndexError
        else:
            while counter < index_:
                current_node = current_node.next
                counter += 1
            return current_node.data


    def __setitem__(self, index_, val_):
        ''' a method which gets an index and a value and sets the given value to the index place in the list '''
        counter = 0
        current_node = self.__head.next
        if index_< 0 or index_> self.__len:
            raise IndexError
        else:
            while counter < index_:
                current_node = current_node.next
                counter += 1
            current_node.data = val_

    def __str__(self):
        ''' a method which defines in which manner the list will be printed to the screen when print function is called '''
        if self.__len == 0:
            return '[]'
        else:
            current_node = self.__head.next
            string = str(current_node.data)
            while current_node != self.__tail.prev:
                current_node = current_node.next
                string += ',' + str(current_node.data)
            return '[' + string + ']'


    def __repr__(self):
        ''' a method which defines in which manner the list will be printed to the screen when repr function is called '''
        if self.__len == 0:
            return '[]'
        else:
            current_node = self.__head.next
            string = str(current_node.data)
            while current_node != self.__tail.prev:
                current_node = current_node.next
                string += ',' + str(current_node.data)
            return '[' + string + ']'

    def __len__(self):
        ''' a method which returns the length of the list'''
        return  self.__len


    def append(self, element_):
        ''' a method which gets an element and adds him to the right side of the list '''
        new_node = self.__Node(None,element_,None)
        self.__len += 1
        if self.__head.next == self.__tail:
            self.__head.next = new_node
            new_node.prev = self.__head
            new_node.next = self.__tail
            self.__tail.prev = new_node
        else:
            previous_node = self.__tail.prev
            previous_node.next = new_node
            new_node.prev = previous_node
            new_node.next = self.__tail
            self.__tail.prev = new_node



    def  pop(self):
        ''' a method which deletes the last element of the list'''
        current_node = self.__tail.prev
        if self.__len == 0:
            raise IndexError
        else:
            current_node.prev.next = self.__tail
            self.__tail.prev = current_node.prev
        return  current_node.data



    def popleft(self):
        '''  a method which deletes the first element from the left side of the list and returns the value '''
        current_node = self.__head.next
        if self.__head.next == self.__tail:
            raise IndexError
        else:
            self.__head.next = current_node.next
            current_node.next.prev = self.__head
            self.__len -= 1
        return  current_node.data


    def remove(self, val_):
        ''' a method which gets a value, starts from the first element and moves through all the list
         and compares the given value with the value stored in each element till it finds the first match
          between the two values and if there is no match with any of the elements a ValueError will be raised '''
        current_node = self.__head
        while current_node != None:
            if current_node.data == val_:
                if current_node.prev != None:
                    current_node.prev.next = current_node.next
                else:
                    self.__head = current_node.next
                if current_node.next != None:
                    current_node.next.prev = current_node.prev
                else:
                    self.__tail = current_node.prev

                self.__len -= 1

            current_node = current_node.next


    def __iter__(self):
        ''' a method which makes the list an iterator and enables us to go through all the list by using a loop'''
        current = self.__head.next
        while current is not self.__tail:
            yield  current.data
            current = current.next


    def reversed(self):
        ''' a method which reverses all the elements in the list '''
        current_node = self.__tail.prev
        if self.__len == 0:
            return '[]'
        else:
            string = str(current_node.data)
            while current_node.prev is not None:
                current_node = current_node.prev
                string += ',' + str(current_node.data)
            return '[' + string[:-5] + ']'


    def __copy__(self):
        ''' a method which creates a new object which is a copy of an existing list '''
        result = Dlist()
        for data in self:
            details = data.__class__(data)
            result.append(details)
        return  result



# ---------------------------------------------------------------------------------------------------------------------

    class __Node(object):
        ''' a nested class which helps to operate and built the Dlist object. 
        each instance of the Node class should contain 3 elements: previous, data and next in order to link
        all the elements in the Dlist '''

        def __init__(self, prev_ = None,data_ = None, next_ = None):
            ''' a method which constructs a Node object each time an instance of the class is created '''
            self.__prev = prev_
            self.__data = data_
            self.__next = next_

        @property
        def data(self):
            ''' this method uses as a getter and returns the value stored in the data part in the relevant node '''
            return  self.__data

        @data.setter
        def data(self, val_):
            ''' this method uses as a setter and enables the user to set values to a node'''
            self.__data = val_

        @property
        def prev(self):
            ''' this method uses as a getter and returns the previous node '''
            return self.__prev

        @prev.setter
        def prev(self, val_):
            ''' this method uses as a setter and enables us to point on other node as a previous node to an
             existing or a new node'''
            self.__prev = val_

        @property
        def next(self):
            ''' this method uses as a getter and returns the next node '''
            return self.__next

        @next.setter
        def next(self, val_):
            ''' this method uses as a setter and enables us to point on other node as a next node to an
             existing or a new node'''
            self.__next = val_

        def __str__(self):
            return str(self.data)

        def __repr__(self):
            return " Node(%s,%S,%S)" % (repr(self.prev),repr(self.data), repr(self.next))



####################   Test   ################################
first_dlist = Dlist(1,2,"f","KEY",6,7,8,9)
print first_dlist
first_dlist.append(101)
print first_dlist

first_dlist.pop()
print first_dlist
first_dlist_copy = first_dlist.__copy__()
print first_dlist_copy

for each in first_dlist:
    print  each
    print "\n"

# testing deep copy, for mutable items inside Dlist: Dlist1 == Dlist2 gives true and Dlist1 is Dlist2 gives false
# the copy complitly independent, any change in Dlist1 doesn't affect Dlist2
print first_dlist == first_dlist_copy , first_dlist == first_dlist_copy, "\n"

reversed_ = first_dlist.reversed()
print reversed_

first_dlist.remove("f")
print first_dlist

res = first_dlist.__getitem__(3)
print res

first_dlist.__setitem__(3,1000000)
print first_dlist

first_dlist.popleft()
print first_dlist





















