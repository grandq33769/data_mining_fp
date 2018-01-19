import json;
import multiprocessing;
import math;
import ast;
import functools;
import re;
null = None;

PROCESSCOUNT = multiprocessing.cpu_count() - 1;

def classof(obj):
    return obj.__class__.__name__;

class Array:
    def __init__(self, *iterable):
        if (len(iterable) == 1):
            self._itr = [*iterable[0]];
        elif (len(iterable) == 0):
            self._itr = [];
        else:
            raise ValueError("Array() constructor only accept one iterable object as the parameter.");
    def __iter__(self):
        return iter(self._itr);
    def __bool__(self):
        return True;
    def __str__(self):
        return self._itr.__str__();
    def __repr__(self):
        return self._itr.__repr__();
    def __getitem__(self, key):
        return self._itr[key] if key < self.length else None;
    def __setitem__(self, key, value):
        if key < self.length:
            self._itr[key] = value;
        elif key == self.length:
            self._itr.append(value);
        else:
            self._itr = [*self._itr, *[None for x in range(0, key-self.length+1)]];
            self._itr[key] = value;
    def __len__(self):
        return self.length;
    def __list__(self):
        return self._itr;
    def push(self, *ele):
        for x in ele:
            self._itr.append(x);
    def filter(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            return Array([*filter(ftn,self._itr)]);
        else:
            return ValueError("filter() only accept a method or function as the parameter.");
    def find(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            allEle = [*filter(ftn,self._itr)];
            if (len(allEle) > 0):
                return allEle[0];
            else:
                return none;
        else:
            return ValueError("find() only accept a method or function as the parameter.");
    def findIndex(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            for i,v in enumerate(self._itr):
                if (ftn(v)):
                    return i;
            return -1;
        else:
            return ValueError("find() only accept a method or function as the parameter.");
    def some(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            return any(self.map(ftn));
        else:
            return ValueError("some() only accept a method or function as the parameter.");
    def every(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            return all(self.map(ftn));
        else:
            return null;
    def map(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            return Array([*map(ftn,self._itr)]);
        else:
            return ftn;
    def join(self, *char):
        if (len(char) == 1):
            return String(str(char[0]).join(self.map(lambda ele: str(ele))));
        elif (len(char) == 0):
            return String("".join(self.map(lambda ele: str(ele))));
        else:
            raise ValueError("join() only accept one string as the parameter.");
    def concat(self, *ary):
        if (len(ary) > 1):
            if (classof(ary[0]) == "Array" or classof(ary[0]) == "list"):
                return Array([*self._itr, *ary[0]]).concat(*ary.slice(1));
            else:
                return Array([*self._itr, ary[0]]).concat(*ary.slice(1));
        elif (len(ary) == 1):
            if (classof(ary[0]) == "Array" or classof(ary[0]) == "list"):
                return Array([*self._itr, *ary[0]]);
            else:
                return Array([*self._itr, ary[0]]);
        else:
            raise ValueError("concat() requires one or more arrays.");
    def includes(self, val):
        return val in self._itr;
    def forEach(self, ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            for i,ele in enumerate(self._itr):
                ftn(ele,i,self);
        else:
            return ValueError("forEach() requires a function/method parameter.");
    def sort(self, ftn):
        self._itr = sorted(self._itr, key=functools.cmp_to_key(ftn));
        return self;
    def slice(self, start, *end):
        if (len(end) == 1):
            return Array(self._itr[start:end[0]]);
        elif (len(end) == 0):
            return Array(self._itr[start:]);
        else:
            raise ValueError("There is only 2 arguemnts, start and end position for the slice function.");
    def indexOf(self, item):
        for i,v in enumerate(self._itr):
            if (v == item):
                return i;
        return -1;
    @property
    def length(self):
        return len(self._itr);

class Map:
    def __init__(self, *iterable):
        if (len(iterable) == 1):
            self._itr = dict(iterable[0]);
        elif (len(iterable) == 0):
            self._itr = dict();
        else:
            raise ValueError("Map() constructor only accept one iterable object as the parameter.");
    def __iter__(self):
        return iter([Array([key,val]) for key,val in self._itr.items()]);
    def __str__(self):
        return self._itr.__str__();
    def __repr__(self):
        return self._itr.__repr__();
    def __bool__(self):
        return True;
    def has(self,key):
        return key in self._itr;
    def set(self,key,value):
        self._itr[key] = value;
    def delete(self,key):
        if (self.has(key)):
            del self._itr[key];
    def clear(self):
        self._itr = dict();
    def get(self,key):
        return self._itr.get(key);
    def keys(self):
        return self._itr.keys();
    def values(self):
        return self._itr.values();
    def entries(self):
        return self._itr.items();
    def forEach(self,ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            for key,val in self._itr.items():
                ftn(val,key,self);
        else:
            return ValueError("forEach() requires a function/method parameter.");
    @property
    def size(self):
        return len(self._itr);

class Set:
    def __init__(self, *iterable):
        if (len(iterable) == 1):
            self._itr = {*iterable[0]};
        elif (len(iterable) == 0):
            self._itr = set();
        else:
            raise ValueError("Set() constructor only accept one iterable object as the parameter.");
    def __iter__(self):
        return self._itr.__iter__();
    def __str__(self):
        return self._itr.__str__();
    def __repr__(self):
        return self._itr.__repr__();
    def __bool__(self):
        return true;
    def has(self,key):
        return key in self._itr;
    def add(self,key):
        self._itr.add(key);
    def delete(self,key):
        if (self.has(key)):
            self._itr.remove(key);
    def clear(self):
        self._itr = set();
    def forEach(self,ftn):
        if (classof(ftn) == "method" or classof(ftn) == "function"):
            for ele in self._itr:
                ftn(ele,ele,self);
        else:
            return ValueError("forEach() requires a function/method parameter.");
    def values(self):
        return iter(self._itr);
    def entries(self):
        return iter([x,x] for x in self._itr);
    @property
    def size(self):
        return len(self._itr);
    
class String:
    def __init__(self, string):
        self._str = str(string);
    def __iter__(self):
        return iter(self._str);
    def __bool__(self):
        return len(self._str) > 0;
    def __str__(self):
        return self._str.__str__();
    def __repr__(self):
        return self._str.__repr__();
    def __hash__(self):
        return hash(self._str);
    def __eq__(self, other):
        return self._str == str(other);
    def __getitem__(self, key):
        if (key < 0 or key >= self.length):
            return null;
        else:
            return String(self._str[key]);
    def __len__(self):
        return self.length;
    def __add__(self,other):
        return String(self._str + str(other));
    def __radd__(self,other):
        return String(str(other) + self._str);
    def __float__(self):
        return float(str(self._str));
    def __int__(self):
        return int(str(self._str));
    def toArray(self):
        return Array([x for x in self._str]).map(lambda x: String(x));
    def trim(self):
        return String(self._str.strip());
    def toLowerCase(self):
        return String(self._str.lower());
    def toUpperCase(self):
        return String(self._str.upper());
    def replace(self, stringORregexp, stringOrFtn):
        replaceType = classof(stringORregexp);
        toType = classof(stringOrFtn);
        if toType == "function" or toType == "method":
            if (replaceType == "SRE_Pattern"):
                return String(re.sub(stringORregexp,lambda mobj: stringOrFtn(mobj.group(0)),self._str));
            elif (replaceType == "str" or replaceType == "String"):
                return String(re.sub(re.compile("[\\"+str(stringOrFtn)+"]"),self._str));
            else:
                raise ValueError("The replacing object must be a string or a regular expression.");
        elif (replaceType == "SRE_Pattern"):
            return String(re.sub(stringORregexp,str(stringOrFtn),self._str));
        elif (replaceType == "str" or replaceType == "String"):
            return String(self._str.replace(str(stringORregexp), str(stringOrFtn), 1));
        else:
            raise ValueError("The replacing object must be a string or a regular expression.");
    def split(self, stringORregexp):
        replaceType = classof(stringORregexp);
        if (replaceType == "SRE_Pattern"):
            return Array(re.split(stringORregexp, self._str)).map(lambda ele: String(ele));
        elif (replaceType == "str" or replaceType == "String"):
            return Array(self._str.split(str(stringORregexp))).map(lambda ele: String(ele));
        else:
            raise ValueError("The delimiter object must be a string or a regular expression.");
    def match(self, regexp):
        if (classof(regexp) == "SRE_Pattern"):
            results = re.findall(regexp, self._str);
            return Array(results) if len(results) > 0 else null;
        else:
            raise ValueError("The match() parameter must be a regular expression.");
    def lastIndexOf(self, string):
        return self._str.rfind(str(string));
    def indexOf(self, string):
        return self._str.find(str(string));
    def slice(self, start, *end):
        if (len(end) == 1):
            return String(self._str[start:end[0]]);
        elif (len(end) == 0):
            return String(self._str[start:]);
        else:
            raise ValueError("There is only 2 arguemnts, start and end position for the slice function.");
    def endsWith(self, string):
        return self._str.endswith(str(string));
    def substr(self, start, end):
        if (start < 0): start = self.length + start;
        if (start < 0): start = 0;
        if (start > self.length): start = self.length;
        if (end < 0): end = 0;
        end = end+start;
        if (end > self.length): end = self.length;
        return String(self._str[start:end]);
    def substring(self, start, end):
        if (start < 0): start = 0;
        if (end < 0): end = 0;
        if (start > self.length): start = self.length;
        if (end > self.length): end = self.length;
        if (start > end): (start, end) = (end, start);
        return String(self._str[start:end]);
    def startsWith(self, string):
        return self._str.startswith(str(string));
    @property
    def length(self):
        return len(self._str);

class JSON:
    @staticmethod
    def parse(string):
        def convertToJS(obj):
            if (classof(obj) == "str"):
                return String(obj);
            elif (classof(obj) == "dict"):
                tmpObj = Map(obj);
                tmpObj.forEach(lambda v,k,m: m.set(k, convertToJS(v)));
                return tempObj;
            elif (classof(obj) == "list"):
                return Array(obj).map(lambda ele: convertToJS(ele));
        if (String(string).trim() == ""):
            return null;
        else:
            preObj = json.loads(str(string));
            return convertToJS(preObj);
    @staticmethod
    def stringify(obj):
        return json.dumps(obj, separators=(',', ':'));

class AST:
    @staticmethod
    def parse(string):
        def convertToJS(obj):
            if (classof(obj) == "str"):
                return String(obj);
            elif (classof(obj) == "dict"):
                tmpObj = Map(obj);
                tmpObj.forEach(lambda v,k,m: m.set(k, convertToJS(v)));
                return tmpObj;
            elif (classof(obj) == "list"):
                return Array(obj).map(lambda ele: convertToJS(ele));
            else:
                return obj;
        if (String(string).trim() == ""):
            return null;
        else:
            preObj = ast.literal_eval(str(string));
            return convertToJS(preObj);
    @staticmethod
    def stringify(obj):
        return repr(obj);

