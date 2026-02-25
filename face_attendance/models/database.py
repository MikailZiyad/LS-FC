import sqlite3
from datetime import datetime
from pathlib import Path
import json
from config.settings import DATABASE_PATH

class DatabaseManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                email TEXT,
                phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                face_image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                attendance_date DATE NOT NULL,
                check_in_time TIME,
                check_out_time TIME,
                status TEXT DEFAULT 'PRESENT',
                confidence_score REAL,
                face_image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_employee(self, employee_id, name, department=None, position=None, email=None, phone=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO employees (employee_id, name, department, position, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (employee_id, name, department, position, email, phone))
            
            conn.commit()
            return True, "Employee added successfully"
        except sqlite3.IntegrityError:
            return False, "Employee ID already exists"
        finally:
            conn.close()
    
    def get_employee_by_id_old(self, employee_id):
        """Get employee by ID (old version - returns tuple)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM employees WHERE employee_id = ?', (employee_id,))
        employee = cursor.fetchone()
        conn.close()
        
        return employee
    
    def get_all_employees(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM employees WHERE is_active = 1 ORDER BY name')
        employees = cursor.fetchall()
        conn.close()
        
        return employees
    
    def add_face_embedding(self, employee_id, embedding, face_image_path=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = json.dumps(embedding).encode('utf-8')
        
        cursor.execute('''
            INSERT INTO face_embeddings (employee_id, embedding, face_image_path)
            VALUES (?, ?, ?)
        ''', (employee_id, embedding_blob, face_image_path))
        
        conn.commit()
        conn.close()
    
    def get_face_embeddings(self, employee_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT embedding FROM face_embeddings WHERE employee_id = ?', (employee_id,))
        embeddings = cursor.fetchall()
        conn.close()
        
        return [json.loads(emb[0].decode('utf-8')) for emb in embeddings]
    
    def record_attendance(self, employee_id, confidence_score=None, face_image_path=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        current_time = datetime.now().strftime('%H:%M:%S')  # Format string untuk SQLite
        
        cursor.execute('''
            SELECT * FROM attendance 
            WHERE employee_id = ? AND attendance_date = ?
        ''', (employee_id, today))
        
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE attendance 
                SET check_out_time = ?, confidence_score = ?, face_image_path = ?
                WHERE employee_id = ? AND attendance_date = ?
            ''', (current_time, confidence_score, face_image_path, employee_id, today))
        else:
            cursor.execute('''
                INSERT INTO attendance (employee_id, attendance_date, check_in_time, confidence_score, face_image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (employee_id, today, current_time, confidence_score, face_image_path))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_today_attendance(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        cursor.execute('''
            SELECT a.*, e.name, e.department 
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE a.attendance_date = ?
            ORDER BY a.check_in_time
        ''', (today,))
        
        attendance = cursor.fetchall()
        conn.close()
        
        # Return as list of dicts for easier access
        result = []
        for att in attendance:
            result.append({
                'employee_id': att[1],
                'attendance_date': att[2],
                'check_in_time': att[3],
                'check_out_time': att[4],
                'status': att[5],
                'confidence_score': att[6],
                'face_image_path': att[7],
                'attendance_time': att[8],  # created_at
                'name': att[9],
                'department': att[10],
            })
        
        return result
    
    def get_recent_attendances(self, limit=10):
        """Get recent attendance records with employee info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.*, e.name, e.department 
            FROM attendance a
            JOIN employees e ON a.employee_id = e.employee_id
            ORDER BY a.created_at DESC
            LIMIT ?
        ''', (limit,))
        
        attendances = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts for easier access
        result = []
        for att in attendances:
            result.append({
                'employee_id': att[1],
                'attendance_date': att[2],
                'check_in_time': att[3],
                'check_out_time': att[4],
                'confidence_score': att[6],
                'name': att[9],
                'department': att[10],
                'attendance_time': att[8]  # created_at
            })
        
        return result
    
    def get_employee_by_id(self, employee_id):
        """Get employee details by employee_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT employee_id, name, department, position, email, phone, created_at, is_active
            FROM employees 
            WHERE employee_id = ? AND is_active = 1
        ''', (employee_id,))
        
        employee = cursor.fetchone()
        conn.close()
        
        if employee:
            return {
                'employee_id': employee[0],
                'name': employee[1],
                'department': employee[2],
                'position': employee[3],
                'email': employee[4],
                'phone': employee[5],
                'created_at': employee[6],
                'is_active': employee[7]
            }
        
        return None

    def get_employee(self, employee_id):
        """Alias for get_employee_by_id for backward compatibility"""
        return self.get_employee_by_id(employee_id)

    def update_employee(self, employee_id, name=None, department=None, position=None, email=None, phone=None, is_active=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        fields = []
        values = []
        if name is not None:
            fields.append("name = ?")
            values.append(name)
        if department is not None:
            fields.append("department = ?")
            values.append(department)
        if position is not None:
            fields.append("position = ?")
            values.append(position)
        if email is not None:
            fields.append("email = ?")
            values.append(email)
        if phone is not None:
            fields.append("phone = ?")
            values.append(phone)
        if is_active is not None:
            fields.append("is_active = ?")
            values.append(1 if is_active else 0)
        fields.append("updated_at = CURRENT_TIMESTAMP")
        set_clause = ", ".join(fields)
        values.append(employee_id)
        cursor.execute(f'''
            UPDATE employees SET {set_clause}
            WHERE employee_id = ?
        ''', tuple(values))
        conn.commit()
        conn.close()
        return True
