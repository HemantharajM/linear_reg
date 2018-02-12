%Load the data 

data = load('./data/housing_price.txt');

%Step1. label the data as input X and output Y
x1 = data(:,1);
x2 = data(:,2);
Y = data(:,3)/1000;

%Step2. plot the data feature inviduallly

%a.number of bedroom vs price 
hp_bed = figure(1);

plot(x2, Y, 'bo');
grid on;
xlabel('Number of bedroom');
ylabel('Price of the House');
title('House price vs Number of bedrooms');

%print(hp_bed, 'bedroom_hp.pdf')

%b.size of house vs price
hp_size = figure(2);

plot(x1, Y, 'r*');
grid on;
xlabel('Size of the House');
ylabel('Price of the House');
title('House price vs Size of the House');

%print(hp_size, 'size_hp.pdf');

m = length(data);

%Step3. Normalize the data or keep data value from 0 to 1
mean_x1 = mean(x1);
std_x2 = std(x1);
X1 = (x1 - mean_x1)./std_x2 ;

mean_x2 = mean(x2);
std_x2 = std(x2);
X2 = (x2 - mean_x2)./std_x2;

x3 = x1.^2;
mean_x3 = mean(x3);
std_x3 = std(x3);
X3 = (x3 - mean_x3)./std_x3;

x4 = x2.^2;
mean_x4 = mean(x4);
std_x4 = std(x4);
X4 = (x4 - mean_x4)./std_x4;

x5 = x1.*x2;
mean_x5 = mean(x5);
std_x5 = std(x5);
X5 = (x5 - mean_x5)./std_x5;

x6 = x1.^3;
mean_x6 = mean(x6);
std_x6 = std(x6);
X6 = (x6 - mean_x6)./std_x6;

x7 = x2.^3;
mean_x7 = mean(x7);
std_x7 = std(x7);
X7 = (x7 - mean_x7) ./ std_x7;
%Step4.select the hypothesis function y_predict =w0 + x1* w1 + x2 * w2 + x1^2 * w3 + x2^2 *w4 + x1 * x2 * w5 + x1^3 * w6

X = [ones(m,1) X1 X2 X3 X4 X5 X6 X7];

W = zeros(8,1);


%Step5. update W value to minimize the cost  J= (Y - Y_predict)^2

iteration = 500;
alpha = 0.01;
lambda = 4650;

j_val = [];

for i = 1:iteration
	Y_predict = X * W;
	J = (Y - Y_predict)' * (Y - Y_predict) + (lambda .* W(4:8)' * W(4:8));
	j_val = [j_val; J];
	
	grad = X' * (Y - Y_predict);

	W(1:3) = W(1:3) + ((alpha/m) * grad(1:3));
	
	W(4:8) = W(4:8) .* (1 - (alpha*lambda)/m) + ((alpha/m) * grad(4:8)); 
end

%Step6. plot the cost value over iteration, to check cost decreasing and when to stop iteration

cost = figure(3);
iter = 1:iteration;

plot(iter, j_val);
xlabel('No of iteration');
ylabel('Cost value');
title('Cost value vs iteration');

print(cost, 'cost_over_iteration.pdf');

%different regularization

%	upto w0 w1 w2 			upto w0 w1 		upto  w0

% cost    1.942  10+5                  1.937 10+5              6.97 10+5       

%  W0     3.38 10+2                    3.381 10+2             338.1759
%  W1     1.036 10+2                   1.068 10+2               1.0118
%  W2     2.339 10-1                  -3.925 10-2               0.5098
%  W3     2.77  10-2                   6.614 10-3               0.9848
%  W4    -3.695 10-2                  -4.748 10-2               0.5296
%  W5     1.568 10-2                  -3.628 10-3               0.9292
%  W6     2.039 10-2                   2.822 10-4               0.9259
%  W7    -3.718 10-2		      -4.818 10-2               0.548
