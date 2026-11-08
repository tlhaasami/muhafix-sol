// Demo contract written for Muhafiz.sol: refunds whose failure is ignored.
// If send() or call() fails, the contract carries on as if the payment worked.
pragma solidity ^0.4.25;

contract TicketRefunds {
    address public organiser;
    mapping(address => uint256) public paid;
    address[] public buyers;

    constructor() public {
        organiser = msg.sender;
    }

    function buy() public payable {
        require(msg.value > 0);
        if (paid[msg.sender] == 0) {
            buyers.push(msg.sender);
        }
        paid[msg.sender] += msg.value;
    }

    function cancelEvent() public {
        require(msg.sender == organiser);
        for (uint256 i = 0; i < buyers.length; i++) {
            address buyer = buyers[i];
            buyer.send(paid[buyer]);
            paid[buyer] = 0;
        }
    }

    function forwardFees(address treasury) public {
        require(msg.sender == organiser);
        treasury.call.value(address(this).balance)();
    }
}
