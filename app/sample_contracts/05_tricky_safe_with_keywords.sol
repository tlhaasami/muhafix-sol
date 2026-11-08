// SPDX-License-Identifier: MIT
// Demo contract written for Muhafiz.sol: uses low-level call (a "suspicious"
// keyword) but correctly: balance is updated first, the result is checked,
// and a reentrancy lock protects the function.
pragma solidity ^0.8.20;

contract SafeVault {
    mapping(address => uint256) private balances;
    bool private locked;

    modifier nonReentrant() {
        require(!locked, "reentrant call");
        locked = true;
        _;
        locked = false;
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) external nonReentrant {
        require(balances[msg.sender] >= amount, "insufficient balance");
        balances[msg.sender] -= amount;
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "transfer failed");
    }

    function balanceOf(address user) external view returns (uint256) {
        return balances[user];
    }
}
