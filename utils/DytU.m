% compute D'_y(U)
function dytu = DytU(U)
    dytu = cat(1, U(end,:,:) - U(1,:,:), -diff(U,1,1));
end
