// Demo contract written for Muhafiz.sol: the "shopkeeper" refund mistake.
// Money is sent BEFORE the balance is set to zero, so an attacker can
// call withdraw() again from inside the payment and be paid repeatedly.
pragma solidity ^0.4.24;

contract CommunityBank {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0);
        require(msg.sender.call.value(amount)());
        balances[msg.sender] = 0;
    }

    function bankBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
