%load the data which has pixel value for river and not a river
river_data = load('./data/river_data');

X = [river_data(:,1:4)];
Y = [river_data(:,5)];

%normalize the data
x1_mean = mean(X);
x1_std = std(X);
X1 = (X - x1_mean)./ x1_std;

X2 = X.^2;
x2_mean = mean(X2);
x2_std = std(X2);
X2 = (X2 - x2_mean) ./ x2_std;

X3 = X.^3;
x3_mean = mean(X3);
x3_std = std(X3);
X3 = (X3 - x3_mean) ./ x3_std;

m = length (X);
%select the hypothesis function h = g(w0 + w1 * x + w2 * x^2 )

X = [ones(m,1) X1 X2 X3];
W = zeros(13,1);

%update the W value
iteration = 500;
alpha = .1;

j_val = [];
for i = 1:iteration
	H = sigmoid(X*W);
	J = sum( - ((Y .* log(H)) +( (1 - Y) .* log(1 - H))));
	j_val = [j_val; J];
	grad = X' * (Y - H);
	
	W = (W )+ ((alpha/m) .* grad);
   
end

%plot the cost function
iter = 1:iteration;
figure(1);
plot(iter, j_val);
%predict the test image
test1 = imread('./data/1.gif');
test2 = imread('./data/2.gif');
test3 = imread('./data/3.gif');
test4 = imread('./data/4.gif');
test = zeros(4,1);
shape_test = size(test1);
rows = shape_test(1);
columns = shape_test(2);
predict = zeros(rows,columns);
for i=1:rows
     for j=1:columns
     	 test = [test1(i,j) test2(i,j) test3(i,j) test4(i,j)];
         predict(i,j) = [1 ,(((double(test)) - x1_mean)./ x1_std) ,(((double(test)).^2 - x2_mean)./ x2_std ), (((double(test)).^3 - x3_mean)./x3_std)] * W;
     end
end
predict = sigmoid(predict);

predict(predict >= 0.5) = 1;
predict(predict < 0.5) = 0;
figure(2);
imshow(predict);
