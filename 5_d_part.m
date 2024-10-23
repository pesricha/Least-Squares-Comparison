% Step 1: Generate the data points
m = 100;  % Number of data points
t = linspace(0, 1, m)';  % 100 points in the interval [0, 1]
f_t = sin(10 * t);  % The function values f(t) = sin(10t)
% Step 2: Construct the Vandermonde matrix for a 14th-degree polynomial
n = 14;  % Degree of the polynomial
A = zeros(m, n+1);  % Initialize the Vandermonde matrix

for j = 0:n
    A(:, j+1) = t.^j;  % A contains powers of t from t^0 to t^14
end
% Step 3: Solve the normal equations using backslash
c = (A' * A) \ (A' * f_t);  % Solve for the coefficients c

p_approx = A_fine * c;  % Polynomial approximation

% Plot the results
figure;
plot(t_fine, sin(10 * t_fine), 'b', 'LineWidth', 2); hold on;
plot(t_fine, p_approx, 'r--', 'LineWidth', 2);
legend('Original function f(t) = sin(10t)', '14th-degree polynomial fit');
title('Least Squares Polynomial Fit using Normal Equations');
xlabel('t');
ylabel('f(t)');

% Step 5: Print the coefficients in the required format
fprintf('Polynomial coefficients (up to 8 decimal places):\n');
for i = 1:length(c)
    if i < length(c)
        fprintf('%.8f, ', c(i));  % Print with comma
    else
        fprintf('%.8f\n', c(i));   % No comma for the last coefficient
    end
end