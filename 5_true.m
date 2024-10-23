% Solve the least squares problem using backslash operator
z = A \ f_t;  % "True" solution coefficients

% Print the true solution in a comma-separated line
fprintf('True solution coefficients (up to 8 decimal places):\n');
for i = 1:length(z)
    if i < length(z)
        fprintf('%.8f, ', z(i));  % Print with comma
    else
        fprintf('%.8f\n', z(i));   % No comma for the last coefficient
    end
end
