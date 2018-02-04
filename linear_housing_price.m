%Load the data and plot the data

data = load('./data/housing_price.txt');

%Step1. label the data as input X and output Y
X = data(:,1:2);
Y = data(:,3);

hp_bed = figure();

plot(X(:,2), Y, 'bo');
grid on;
xlabel('Number of bedroom');
ylabel('Price of the House');
title('House price vs Number of bedrooms');

%print(hp_bed, 'bedroom_hp.pdf')

hp_size = figure();

plot(X(:,1), Y, 'r*');
grid on;
xlabel('Size of the House');
ylabel('Price of the House');
title('House price vs Size of the House');

%print(hp_size, 'size_hp.pdf');
