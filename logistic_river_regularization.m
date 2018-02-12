%load the data which has pixel value for river and not a river
river_data = load('./data/my_datar3.txt');
not_river_data = load('./data/my_datanr3.txt');

X = [river_data(:,1); not_river_data(:,1)];
Y = [river_data(:,2); not_river_data(:,2)];

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
%select the hypothesis function h = g(w0 + w1 * x + w2 * x^2+ w3 * x^3 )

X = [ones(m,1) X1 X2 X3];
W = zeros(4,1);

%update the W value
iteration = 20000;
alpha = .1;
lambda = 1490;
j_val = [];
for i = 1:iteration
	H = sigmoid(X*W);
	J = sum( - ((Y .* log(H)) +( (1 - Y) .* log(1 - H)))) + (lambda * (W(4)' * W(4)));
	j_val = [j_val; J];
	grad = X' * (Y - H);
	
	W(1:3) = W(1:3) + ((alpha/m) .* grad(1:3));
	W(4) = W(4) .* (1 - (alpha*lambda)/m) + ((alpha/m) .* grad(4));
   
end

%plot the cost function
iter = 1:iteration;
figure(1);
plot(iter, j_val);
%predict the test image
test = imread('./data/3.gif');
shape_test = size(test);
rows = shape_test(1);
columns = shape_test(2);
predict = zeros(rows,columns);
for i=1:rows
     for j=1:columns
         predict(i,j) = [1 ,((double(test(i,j)) - x1_mean)./ x1_std) ,((double(test(i,j))^2 - x2_mean)./ x2_std ), ((double(test(i,j)).^3 -x3_mean) ./ x3_std) ] * W;
     end
end
predict = sigmoid(predict);

predict(predict >= 0.5) = 1;
predict(predict < 0.5) = 0;
figure(2);
imshow(predict);

%plot the boundary conditions
minimum_pixel_value = 1;
maximum_pixel_value = max(max(test));
pix = minimum_pixel_value: maximum_pixel_value;

boundary = zeros(minimum_pixel_value, maximum_pixel_value);

for i = pix
	boundary(1,i) = [1 (i - x1_mean) / x1_std  ((i*i) - x2_mean) / x2_std ((i*i*i) - x3_mean) / x3_std] * W;
end

figure(3);
grid on;
plot(pix, boundary);
