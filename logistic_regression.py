import numpy as np 

#hàm sigmoid
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
    
# khởi tạo vectơ trọng số và hệ số bias
def initialize_with_zeros(dim):
    w = np.zeros(dim)
    b = 0
    return w, b
    
# Tính toán hàm mất mát và Gradient
def propagate(w, b, x, y):
    # số bản ghi dữ liệu
    m = x.shape[0]
    # Tính toán hàm kích hoạt sigmoid
    A = sigmoid(np.dot(x, w) + b)
    # Tính toán hàm mất mát 
    loss = (-1 / m) * (np.dot(y, np.log(A).T) + np.dot((1 - y), np.log(1 - A).T))
    #loss = np.squeeze(loss) 
    # Tính toán Gradident
    dw = (1 / m) * np.dot((A - y).T, x)
    db = (1 / m) * np.sum(A - y)
    grad = {'dw': dw, 'db': db}
    
    return grad, loss

# Hàm cập nhật tham số, tối ưu hóa bắng cách chạy Gradient Descent
def optimize(w, b, x, y, number_of_iterations, learning_rate, print_loss = False):
    # danh sách lưu trữ lịch sử hàm mất mát
    loss_history = []
    
    # lặp lại và tối ưu hóa các tham số
    for i in range(number_of_iterations):
        # tính hàm mất mát và gradient
        grad, loss = propagate(w, b, x, y)
        # lấy giá trị
        dw = grad['dw']
        db = grad['db']
        
        #khi tham số không thay đổi quá nhiều thì ta dừng vòng lặp
        if (np.linalg.norm(dw)**2 + np.linalg.norm(db)**2) < 1e-6:
            break;
            
        #nếu không ta cập nhật tham số
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # mất mát sau mỗi 500 lần lặp
        if i % 500 == 0:
            loss_history.append(loss)
            
        # in ra mất mát sau mỗi 500 lần lặp
        if print_loss and i % 500 == 0:
            print('Loss after {0}: {1}'.format(i, loss))
    
    # lưu lại tham số đã cập nhật và gradient
    params = {'w': w, 'b': b}
    grad = {'dw': dw, 'db': db}
            
    return params, grad, loss_history

# hàm dự đoán
def predict(w, b, x):
    # Lấy số bản ghi dữ liệu đầu vào
    m = x.shape[0]
    
    # Tạo vectơ lưu kết quả
    y_prediction = np.zeros(m)    
    w = w.reshape(x.shape[1], 1)
    
    # Tính toán xác suất
    A = sigmoid(np.dot(x, w) + b)
    
    # Chuyển xác suất sang 0 và 1
    for i in range(A.shape[0]):
        if A[i, 0] >= 0.5:
            y_prediction[i] = 1
        else:
            y_prediction[i] = 0
    
    return y_prediction
    
# Tạo mô hình
def model(x_train, y_train, x_test, y_test, number_of_iterations = 5000, learning_rate = 0.05, print_loss = False):
    # khởi tạo tham số (trọng số w và hệ số bias b)
    w, b = initialize_with_zeros(x_train.shape[1])
    
    # Tối ưu hóa với Gradient Descent
    params, grad, loss_history = optimize(w, b, x_train, y_train, number_of_iterations, learning_rate, print_loss)
    # lấy trọng số đã tính
    w = params['w']
    b = params['b']
    
    # dữ đoán cho tập train và test
    y_prediction_train = predict(w, b, x_train)
    y_prediction_test = predict(w, b, x_test)
    
    #một số thông số mô hình
    d = {'y_prediction_train' : y_prediction_train,
         'y_prediction_test': y_prediction_test,
         'w' : w, 
         'b' : b,
         'learning_rate' : learning_rate,
         'number_of_iterations': number_of_iterations}
    
    return d