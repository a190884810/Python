# Lab Assignment for Tiancheng Xie and Muhammad Zubair

#Q1
def str_searching(str_a):
    str_b = str_a.strip()
    is_repeat = False
    count = 1
    for i in str_b:
        for j in str_b[count::]:
            if(j.upper() == i.upper()):
                count += 1
                is_repeat = True
                break
        if(is_repeat == True):
            is_repeat = False
        else:
            break
    print(i)


#Q2

with open('file1.txt', 'r+') as f1:
    with open('file2.txt', 'r') as f2:
        for line1 in f1:
            list1 = line1.split() #store file1 words into list1
        for line2 in f2:
            list2 = line2.split() #store file2 words into list2
        for i in list2:
            while((i in list1) or ((i[0].upper()+i[1:]) in list1)):  #if words in file2 in file1
                list1.remove(i) if i in list1 else list1.remove(i[0].upper()+i[1:])
        str_file1 = " ".join(list1)  #combine list 1
        
        f1.seek(0) #point to 0 position
        f1.truncate() #clear the file
        f1.write(str_file1)


#Q3

def symmetrical_difference(seta,setb):
    return ((seta | setb) - setb)



#Q4
class Hospital: #first class(a)
    def __init__(self): #(b)
        self.title = "Welcome to the Hospital Admission System"

class Patient(Hospital): #second class(a),inheritance(c)
    list_of_patient = []   #at least one private data(f)
    count = 0              #counting the number of patients
    def __init__(self,name): #initial constructor (b)
        Hospital.__init__(self)   #use of self(e)
        self.patient_name = name
        Patient.count += 1
        Patient.list_of_patient.append(self.patient_name)
    def display_patient(self):
        print(Patient.list_of_patient)
    def __del__(self):
        print("deleted")

class Serious_Patient(Patient): #multiple inheritance
    list_of_serious_patient = []
    serious_count = 0
    def __init__(self,name):
        Patient.__init__(self,name)
        self.name = name
        Serious_Patient.count += 1
        Serious_Patient.list_of_serious_patient.append(self.name)
    def display_patient(self):
        print(Serious_Patient.list_of_serious_patient)
        

class Doctor(Hospital): #third class(a)
    list_of_doctor = []
    count = 0 #counting the number of doctors
    def __init__(self,name,department): #(b)
        super(Doctor,self).__init__()  #super call(d)
        self.doctor_name = name
        self.doctor_department = department
        Doctor.count +=1
        Doctor.list_of_doctor.append(self.doctor_name)
    def display_doctor(self):
        print(Doctor.list_of_doctor)

class Book(Patient): #fourth class(a)
    list_of_line = []
    def __init__(self,name): #(b)
        Patient.__init__(self,name)
        
        Book.list_of_line.append(self.patient_name)
    def remove(self):
        Book.list_of_line.remove(self.patient_name)
        __del__(self)
    def display_list(self):
        print(self.__class__.list_of_line)

class Nurse(Hospital): #fifth class(a)
    list_of_nurse = []
    count = 0 #counting the number of nurses
    def __init__(self,name): #(b)
        Hospital.__init__(self)
        self.nurse_name = name
        Nurse.count +=1
        Nurse.list_of_nurse.append(self.nurse_name)
    def display_nurse(self):
        print(Nurse.list_of_nurse)
# creating instances

a = Hospital()
b = Book("Jack")
b.display_list()
f = Patient("Rat")
f.display_patient()
c = Serious_Patient("Tom")
c.display_patient()
d = Doctor("Hanson","PE")
d.display_doctor()
e = Nurse("Jenny")
e.display_nurse()
del f



#Q5

import pandas as pd

url = "https://www.fantasypros.com/nfl/reports/leaders/qb.php?year=2015"
data = pd.read_html(url)[0] #crazy pandas
print(data)
with open('table.txt','w') as f: #write into table.txt file
    f.write(str(data))
    
