% Combined MATLAB Code for 14th Degree Polynomial Fit

% Step 1: Generate Data Points
m = 100; % Number of data points
t = linspace(0, 1, m)'; % Column vector of t values
f_t = sin(10 * t); % Compute f(t)

% Step 2: Construct the Design Matrix
X = zeros(m, 15); % 15 columns for degree 0 to 14
for j = 0:14
    X(:, j + 1) = t .^ j; % Fill in each column
end

% Step 3: Solve the Normal Equations
c = (X' * X) \ (X' * f_t); % Coefficients of the polynomial

% Print the coefficients
fprintf('Coefficients of the 14th Degree Polynomial Fit:\n');
for j = 1:length(c)
    fprintf('c(%d) = %.8f\n', j - 1, c(j)); % Print each coefficient
end

% Step 4: Create the Polynomial Function
polynomial_fit = @(t) c(1) + c(2) * t + c(3) * t.^2 + c(4) * t.^3 + ...
                      c(5) * t.^4 + c(6) * t.^5 + c(7) * t.^6 + ...
                      c(8) * t.^7 + c(9) * t.^8 + c(10) * t.^9 + ...
                      c(11) * t.^10 + c(12) * t.^11 + c(13) * t.^12 + ...
                      c(14) * t.^13 + c(15) * t.^14;

% Step 5: Evaluate and Plot the Results
f_fit = polynomial_fit(t); % Evaluate the polynomial at the data points

% Plot the original function and the polynomial fit
figure;
plot(t, f_t, 'b-', 'DisplayName', 'f(t) = sin(10t)'); % Original function
hold on;
plot(t, f_fit, 'r--', 'DisplayName', '14th Degree Polynomial Fit'); % Polynomial fit
xlabel('t');
ylabel('f(t)');
title('Least Squares Polynomial Fit');
legend;
grid on;
hold off;

% Print machine epsilon
machine_epsilon = eps; % Machine epsilon in MATLAB
fprintf('Machine epsilon: %.8e\n', machine_epsilon);
