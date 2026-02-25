from models.database import DatabaseManager
from datetime import datetime

class Employee:
    def __init__(self, employee_id, name, department=None, position=None, email=None, phone=None):
        self.employee_id = employee_id
        self.name = name
        self.department = department
        self.position = position
        self.email = email
        self.phone = phone
        self.created_at = datetime.now()
        self.is_active = True
    
    def save(self):
        db = DatabaseManager()
        return db.add_employee(self.employee_id, self.name, self.department, 
                           self.position, self.email, self.phone)
    
    @staticmethod
    def get_by_id(employee_id):
        db = DatabaseManager()
        return db.get_employee_by_id(employee_id)
    
    @staticmethod
    def get_all():
        db = DatabaseManager()
        return db.get_all_employees()